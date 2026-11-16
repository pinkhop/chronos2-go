package network

import (
	"context"
	"fmt"
)

// defaultBlockTokens is the target number of tokens processed together by one
// sublayer step. Every buffer other than the residual stream is sized by it,
// so a pass's memory is about Rows × Tokens × ModelWidth floats plus a fixed
// amount, however large the batch; 4096 tokens also give each parallel
// product enough work to keep every core busy.
const defaultBlockTokens = 4096

// queryBlock is the number of query tokens in one time-attention work unit.
// It bounds each worker's score matrix to queryBlock × Tokens values and
// gives short rows enough units to spread across workers.
const queryBlock = 64

// keyAlignment is the multiple to which time attention pads each row's key
// count. Padding to a multiple of the widest vector (16 float32 lanes on
// AVX-512) gives the attention-weighted value product a reduction length
// with no tail and keeps every score row and transposed value row on the
// same vector alignment as its buffer.
const keyAlignment = 16

// forwardPass holds the dimensions and working memory of one forward pass.
// It is created per call and used by that call alone; parallel steps write
// disjoint parts of its buffers.
type forwardPass struct {
	// network supplies the weights and config.
	network *Network
	// config is the network's config.
	config Config
	// batch is the input; it is only read.
	batch Batch
	// observe receives intermediate stages; it is never nil.
	observe StageObserver
	// observing reports whether the caller supplied an observer, which makes
	// the pass compute every stage, including all final hidden states.
	observing bool
	// workers is the parallelism limit, at least 1.
	workers int

	// rows is R; contextPatches is N; outputPatches is P.
	rows, contextPatches, outputPatches int
	// tokens is S = N + [REG] + P, the encoder positions per row.
	tokens int
	// paddedTokens is S rounded up to a multiple of keyAlignment, the key
	// count of each row's time attention.
	paddedTokens int
	// futureOffset is the token index of the first output patch in a row.
	futureOffset int
	// totalTokens is R × S.
	totalTokens int
	// innerWidth is Heads × HeadWidth.
	innerWidth int
	// timeBlockRows is the number of rows per time-attention block.
	timeBlockRows int
	// groupBlockTokens is the number of token positions per group-attention
	// block.
	groupBlockTokens int
	// blockCapacity is the most tokens any block holds.
	blockCapacity int
	// singletonGroups reports that every row is alone in its group, which
	// lets group attention skip queries, keys, and scores.
	singletonGroups bool

	// locations and scales hold each row's normalization statistics.
	locations, scales []float32
	// hidden holds the residual stream, [R × S × ModelWidth].
	hidden []float32
	// blockInput holds a block's normalized tokens and sublayer outputs,
	// [blockCapacity × ModelWidth].
	blockInput []float32
	// query, key, value, and attention hold a block's projected heads,
	// [blockCapacity × innerWidth].
	query, key, value, attention []float32
	// headKey holds a time-attention block's rotated keys by row and head,
	// [timeBlockRows × Heads × paddedTokens × HeadWidth], and headValue its
	// values transposed, [timeBlockRows × Heads × HeadWidth × paddedTokens].
	// Their padding keys and values are zero and never written, so padded
	// scores are exactly zero before masking and padded values contribute
	// exactly zero to each output.
	headKey, headValue []float32
	// activations holds a block's feed-forward or residual-block hidden
	// activations, [blockCapacity × FeedForwardWidth].
	activations []float32
	// residual holds a group-attention block's gathered tokens, or a
	// residual block's output and skip projection side by side,
	// [blockCapacity × 2 × the widest of ModelWidth and residual-block outputs].
	residual []float32
	// keyObserved marks tokens that may be attended to, [R × S].
	keyObserved []bool
	// timeMask holds additive time-attention key masks, [R × paddedTokens]:
	// zero or the most negative finite float32 for each real key, and
	// negative infinity for each padding key, so padding keys receive
	// exactly zero weight even in a row whose real keys are all masked.
	timeMask []float32
	// sameGroup marks row pairs in one group, [R × R].
	sameGroup []bool
	// cos and sin are the rotary tables, [S × HeadWidth].
	cos, sin []float32
	// scratch holds idle worker scratch; its capacity is workers, so a
	// returning worker never blocks.
	scratch chan *workerScratch
}

// workerScratch is one worker's private working memory, reused across the
// units it processes during a pass.
type workerScratch struct {
	// query holds up to queryBlock gathered query vectors (or every row's
	// in group attention), and output the matching attention outputs.
	query, output []float32
	// key holds one head's gathered group-attention keys and
	// valueTransposed its values transposed to [HeadWidth × R].
	key, valueTransposed []float32
	// scores holds one unit's attention matrix.
	scores []float32
	// groupMask holds one token's additive group mask, R × R values.
	groupMask []float32
	// tile holds one narrow projection tile before it is copied into place.
	tile []float32
}

// newForwardPass computes the pass's dimensions and allocates its buffers;
// worker scratch is allocated on first use. tuning organizes the work;
// results do not depend on it.
func newForwardPass(network *Network, batch Batch, workers int, observe StageObserver, tuning passTuning) *forwardPass {
	blockTokens := tuning.blockTokens
	config := network.config
	p := &forwardPass{
		network: network, config: config, batch: batch, observe: observe, observing: observe != nil, workers: workers,
		rows: batch.Rows, outputPatches: batch.OutputPatches,
		contextPatches: (batch.ContextLength + config.PatchSize - 1) / config.PatchSize,
		innerWidth:     config.Heads * config.HeadWidth,
		scratch:        make(chan *workerScratch, workers),
	}
	if observe == nil {
		p.observe = func(string, []float32) {}
	}
	p.futureOffset = p.contextPatches
	if config.UseRegToken {
		p.futureOffset++
	}
	p.tokens = p.futureOffset + p.outputPatches
	p.paddedTokens = (p.tokens + keyAlignment - 1) / keyAlignment * keyAlignment
	p.totalTokens = p.rows * p.tokens
	p.timeBlockRows = min(p.rows, max(1, blockTokens/p.tokens))
	p.groupBlockTokens = min(p.tokens, max(1, blockTokens/p.rows))
	p.blockCapacity = max(p.timeBlockRows*p.tokens, p.groupBlockTokens*p.rows, min(blockTokens, p.totalTokens))
	p.singletonGroups = !tuning.generalGroups && distinct(batch.GroupIDs)
	p.allocate()
	return p
}

// distinct reports whether no two group IDs are equal.
func distinct(groupIDs []int) bool {
	seen := make(map[int]struct{}, len(groupIDs))
	for _, group := range groupIDs {
		if _, ok := seen[group]; ok {
			return false
		}
		seen[group] = struct{}{}
	}
	return true
}

// allocate creates the pass's shared buffers.
func (p *forwardPass) allocate() {
	capacity, width := p.blockCapacity, p.config.ModelWidth
	p.locations = make([]float32, p.rows)
	p.scales = make([]float32, p.rows)
	p.hidden = make([]float32, p.totalTokens*width)
	p.blockInput = make([]float32, capacity*width)
	p.query = make([]float32, capacity*p.innerWidth)
	p.key = make([]float32, capacity*p.innerWidth)
	p.value = make([]float32, capacity*p.innerWidth)
	p.attention = make([]float32, capacity*p.innerWidth)
	p.activations = make([]float32, capacity*p.config.FeedForwardWidth)
	p.residual = make([]float32, capacity*2*max(width, p.network.outputBlock.outputWidth))
	p.keyObserved = make([]bool, p.totalTokens)
	headValues := p.timeBlockRows * p.config.Heads * p.paddedTokens * p.config.HeadWidth
	p.headKey = make([]float32, headValues)
	p.headValue = make([]float32, headValues)
	p.timeMask = make([]float32, p.rows*p.paddedTokens)
	p.sameGroup = make([]bool, p.rows*p.rows)
}

// newScratch allocates one worker's scratch sized for this pass.
func (p *forwardPass) newScratch() *workerScratch {
	members := max(p.tokens, p.rows)
	headValues := members * p.config.HeadWidth
	widest := max(p.config.FeedForwardWidth, p.config.ModelWidth, p.innerWidth, p.network.outputBlock.outputWidth)
	return &workerScratch{
		query:           make([]float32, headValues),
		output:          make([]float32, headValues),
		key:             make([]float32, p.rows*p.config.HeadWidth),
		valueTransposed: make([]float32, p.rows*p.config.HeadWidth),
		scores:          make([]float32, max(queryBlock*p.paddedTokens, p.rows*p.rows)),
		groupMask:       make([]float32, p.rows*p.rows),
		tile:            make([]float32, projectionRowTile*widest),
	}
}

// run executes every stage and assembles the output, checking ctx between
// stages and before every sublayer of every encoder layer.
func (p *forwardPass) run(ctx context.Context, wantHidden bool) (Output, error) {
	if err := ctx.Err(); err != nil {
		return Output{}, interrupted(err)
	}
	p.prepareMasks()
	p.cos, p.sin = rotaryTables(p.tokens, p.config.HeadWidth, p.config.RopeTheta)
	p.embed(p.normalize())
	for index, layer := range p.network.layers {
		if err := p.runLayer(ctx, index, layer); err != nil {
			return Output{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return Output{}, interrupted(err)
	}
	output := Output{
		Rows:      p.rows,
		Quantiles: len(p.config.Quantiles),
		Steps:     p.outputPatches * p.config.PatchSize,
		Locations: p.locations,
		Scales:    p.scales,
	}
	var forecastTokens []float32
	if wantHidden || p.observing {
		var hidden []float32
		hidden, forecastTokens = p.finalNormAll()
		if wantHidden {
			output.Tokens, output.Width, output.Hidden = p.tokens, p.config.ModelWidth, hidden
		}
	} else {
		forecastTokens = p.finalNormForecast()
	}
	output.Values = p.decode(forecastTokens)
	return output, nil
}

// runLayer applies encoder layer index's time attention, group attention, and
// feed-forward sublayers to the residual stream, checking ctx before each one.
// A sublayer of the base model over an 8192-step context takes long enough
// that checking only between layers would leave cancellation waiting for a
// whole layer. It returns an error wrapping ctx.Err() when ctx has ended,
// leaving the residual stream partly updated.
func (p *forwardPass) runLayer(ctx context.Context, index int, layer layerWeights) error {
	sublayers := [...]func(){
		func() { p.timeAttention(layer.time) },
		func() { p.groupAttention(layer.group) },
		func() { p.feedForward(layer.feedForward) },
	}
	for _, sublayer := range sublayers {
		if err := ctx.Err(); err != nil {
			return interrupted(err)
		}
		sublayer()
	}
	p.observe(layerStageName(index), p.hidden)
	return nil
}

// interrupted wraps a context error so callers can match it with errors.Is.
func interrupted(err error) error {
	return fmt.Errorf("network: forward pass interrupted: %w", err)
}
