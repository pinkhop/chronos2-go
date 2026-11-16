package chronos2

import (
	"context"
	"fmt"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// embedOutputPatches is the number of masked output patches an embedding
// pass appends, matching the reference pipeline's embed().
const embedOutputPatches = 1

// Embedding holds the vectors the model builds internally while reading one
// history, arranged so that series that behave alike get similar vectors.
// The model reads a history in patches of 16 consecutive steps and produces
// one vector per patch, one summary vector for an extra position it appends
// after the last patch, and one vector for the empty future it was asked to
// fill. Every vector has [Model.EmbeddingDimension] elements, and vectors
// from different models are not comparable.
//
// Because the model normalizes each history by its mean and standard
// deviation before reading it, embeddings describe a history's shape, not its
// level: a series and the same series times 1000 get essentially the same
// vectors.
// Location and Scale report what was removed.
//
// An Embedding is immutable and safe for concurrent use; every accessor
// returns a newly allocated slice the caller may modify.
type Embedding struct {
	// states holds every encoder state, one vector of width values each,
	// owned by this Embedding and never modified.
	states [][]float32

	// patches is the number of context patch states at the start of states.
	patches int

	// summary is the index of the [REG] state in states, or -1 when the model
	// has no [REG] token.
	summary int

	// location is the mean of the history's observed values within the
	// context window.
	location float64

	// scale is the standard deviation of the history's observed values within
	// the context window.
	scale float64
}

// Summary returns one vector describing the whole history: the model's state
// at the extra position it appends after the last patch (called the [REG]
// token), which draws on every patch. Use it for similarity search,
// clustering, and features for other models, comparing vectors with cosine
// similarity. Histories of any length yield vectors of the same length that
// can be compared, but each vector reflects only the context the model saw,
// so a 50-step history and a 5,000-step history of the same series can
// differ simply because one shows the model far more. Chronos-2 is trained
// only to forecast, so if Summary separates a set of series poorly, the
// element-wise mean of the Patches vectors is the usual alternative. It
// returns nil for a checkpoint without a [REG] token and for the zero
// Embedding.
func (e Embedding) Summary() []float32 {
	if e.summary < 0 || e.summary >= len(e.states) {
		return nil
	}
	return slices.Clone(e.states[e.summary])
}

// Patches returns one vector per 16-step patch of the history, oldest first,
// for comparing or locating segments within a history, such as finding where
// a pattern occurs or where behavior changes. A history of n steps, after
// trimming to the context window, yields ceil(n/16) patches; when n is not a
// multiple of 16, the oldest patch is partly padding.
func (e Embedding) Patches() [][]float32 {
	return cloneVectors(e.states[:e.patches])
}

// States returns every vector the model produced, in order: the Patches
// vectors, the Summary vector, and the vector for the empty future patch. It
// matches the embed() output of Chronos-2's own Python pipeline; most callers
// want Summary or Patches.
func (e Embedding) States() [][]float32 {
	return cloneVectors(e.states)
}

// Location returns the mean of the history's observed values within the
// context window, which the model subtracts before reading the history. Use
// it with Scale when comparisons must account for a series' level.
func (e Embedding) Location() float64 { return e.location }

// Scale returns the standard deviation of the history's observed values
// within the context window, which the model divides by before reading the
// history. Use it with Location when
// comparisons must account for a series' magnitude.
func (e Embedding) Scale() float64 { return e.scale }

// cloneVectors deep-copies vectors.
func cloneVectors(vectors [][]float32) [][]float32 {
	cloned := make([][]float32, len(vectors))
	for i, vector := range vectors {
		cloned[i] = slices.Clone(vector)
	}
	return cloned
}

// Embed returns the model's internal vectors for one history (see
// [Embedding]), costing about as much as one forecast. The history follows
// the rules of [Model.Forecast]: oldest first, NaN for a missing
// observation, and trimmed to its most recent [Model.ContextLength] steps or
// the [WithContextLength] value. It accepts WithContextLength and
// [WithMaxParallelism].
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first.
func (m *Model) Embed(ctx context.Context, history []float64, opts ...EmbedOption) (Embedding, error) {
	settings, err := collectEmbedOptions(opts).resolve(m.spec.contextLength)
	if err != nil {
		return Embedding{}, err
	}
	if err := checkHistory(history, len(history)); err != nil {
		return Embedding{}, fmt.Errorf("history: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return Embedding{}, fmt.Errorf("chronos2: embed canceled: %w", err)
	}

	// One history forms one row with an all-NaN future patch.
	batch := batching.NewPlan([]batching.Task{{Targets: [][]float64{history}}}, batching.Options{
		Horizon:       embedOutputPatches * m.spec.outputPatchSize,
		ContextLength: settings.contextLength,
		BatchSize:     1,
	}).Batch(0)
	output, err := m.forward.Forward(ctx, network.Batch{
		Rows:             batch.Rows,
		ContextLength:    batch.ContextLength,
		Context:          batch.Context,
		GroupIDs:         batch.GroupIDs,
		OutputPatches:    embedOutputPatches,
		FutureCovariates: batch.Future,
	}, forwardOptions{parallelism: settings.parallelism, hidden: true})
	if err != nil {
		return Embedding{}, wrapEngineError(err)
	}
	return m.newEmbedding(output, batch.ContextLength)
}

// newEmbedding builds an Embedding from a one-row pass over a context of
// contextLength steps, returning an error wrapping ErrInvalidCheckpoint when
// the hidden states' shape disagrees with the model.
func (m *Model) newEmbedding(output network.Output, contextLength int) (Embedding, error) {
	patches := (contextLength + m.spec.outputPatchSize - 1) / m.spec.outputPatchSize
	summary, tokens := -1, patches+embedOutputPatches
	if m.spec.regToken {
		summary, tokens = patches, tokens+1
	}
	width := m.spec.embeddingDimension
	if output.Rows != 1 || output.Tokens != tokens || output.Width != width || len(output.Hidden) != tokens*width ||
		len(output.Locations) != 1 || len(output.Scales) != 1 {
		return Embedding{}, fmt.Errorf("%w: network returned %d rows of %d×%d hidden states, want 1 row of %d×%d",
			ErrInvalidCheckpoint, output.Rows, output.Tokens, output.Width, tokens, width)
	}
	states := make([][]float32, tokens)
	for token := range states {
		states[token] = slices.Clone(output.Hidden[token*width : (token+1)*width])
	}
	return Embedding{
		states:   states,
		patches:  patches,
		summary:  summary,
		location: float64(output.Locations[0]),
		scale:    float64(output.Scales[0]),
	}, nil
}
