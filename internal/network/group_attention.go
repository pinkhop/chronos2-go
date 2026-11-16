package network

import (
	"github.com/pinkhop/chronos2-go/internal/kernel"
	"github.com/pinkhop/chronos2-go/internal/parallel"
)

// groupAttention applies the pre-norm group self-attention sublayer: at each
// token position, every row attends across the rows of its group whose token
// there is unmasked. Rows form an unordered set, so no position embedding
// applies. Token positions are processed in blocks: each block's tokens are
// gathered from every row into blockInput (row-major, then token), updated,
// and added back.
func (p *forwardPass) groupAttention(weights attentionWeights) {
	width := p.config.ModelWidth
	for firstToken := 0; firstToken < p.tokens; firstToken += p.groupBlockTokens {
		count := min(p.groupBlockTokens, p.tokens-firstToken)
		tokens := p.rows * count
		gathered := p.residual[:tokens*width]
		for row := range p.rows {
			start := (row*p.tokens + firstToken) * width
			copy(gathered[row*count*width:(row+1)*count*width], p.hidden[start:start+count*width])
		}
		if p.singletonGroups {
			p.projectValues(weights, gathered, tokens)
			p.forEachUnit(count, func(scratch *workerScratch, position int) {
				p.singletonGroupToken(scratch, firstToken+position, position, count)
			})
		} else {
			p.projectHeads(weights, gathered, tokens)
			p.forEachUnit(count, func(scratch *workerScratch, position int) {
				p.groupAttentionToken(scratch, firstToken+position, position, count)
			})
		}
		p.finishAttention(weights, gathered, tokens)
		parallel.For(p.rows, p.workers, func(start, end int) {
			for row := start; row < end; row++ {
				first := (row*p.tokens + firstToken) * width
				copy(p.hidden[first:first+count*width], gathered[row*count*width:(row+1)*count*width])
			}
		})
	}
}

// groupAttentionToken computes every head's group attention at token, which
// is block position position of a block holding count positions per row. A
// key row is visible to a query row when both share a group and the key's
// token is unmasked; when no key is visible, the additive mask leaves every
// score equal, so the query attends uniformly to all rows, as in the
// reference.
func (p *forwardPass) groupAttentionToken(scratch *workerScratch, token, position, count int) {
	rows, headWidth := p.rows, p.config.HeadWidth
	mask := scratch.groupMask[:rows*rows]
	for queryRow := range rows {
		for keyRow := range rows {
			visible := p.sameGroup[queryRow*rows+keyRow] && p.keyObserved[keyRow*p.tokens+token]
			mask[queryRow*rows+keyRow] = 0
			if !visible {
				mask[queryRow*rows+keyRow] = maskedScore
			}
		}
	}
	query, key := scratch.query[:rows*headWidth], scratch.key[:rows*headWidth]
	valueTransposed := scratch.valueTransposed[:rows*headWidth]
	scores, output := scratch.scores[:rows*rows], scratch.output[:rows*headWidth]
	for head := range p.config.Heads {
		p.gatherHead(query, p.query, position, count, rows, head)
		p.gatherHead(key, p.key, position, count, rows, head)
		p.gatherHeadTransposed(valueTransposed, p.value, position, count, rows, rows, head)
		kernel.MatMulTransposed(scores, query, key, rows, headWidth, rows)
		kernel.Add(scores, scores, mask)
		kernel.SoftmaxRows(scores, rows, rows)
		kernel.MatMulTransposed(output, scores, valueTransposed, rows, rows, headWidth)
		p.scatterHead(output, position, count, rows, head)
	}
}

// projectValues normalizes tokens rows of input into blockInput and projects
// them to values only; singleton groups need no queries or keys.
func (p *forwardPass) projectValues(weights attentionWeights, input []float32, tokens int) {
	width := p.config.ModelWidth
	normed := p.blockInput[:tokens*width]
	p.rmsNorm(normed, input, weights.norm, tokens)
	p.project(projection{
		out: p.value[:tokens*p.innerWidth], in: normed, weight: weights.value,
		rows: tokens, inWidth: width, outWidth: p.innerWidth,
	})
}

// singletonGroupToken computes group attention at token when every row is
// alone in its group, without queries, keys, or scores; scratch is the
// calling worker's private memory. A row whose token is unmasked sees only
// its own key, so the softmax gives it weight exactly 1 and the result is its
// own value. A masked row sees no key, so the additive mask makes every score
// equal and the softmax gives each row weight 1/R; every masked row therefore
// gets the same uniform average of all rows' values, computed once by
// uniformGroupAverage. Both cases reproduce the general computation bit for
// bit whenever |q·k| is far below the mask magnitude (about 3.4e38), which
// holds for any normalized input.
func (p *forwardPass) singletonGroupToken(scratch *workerScratch, token, position, count int) {
	anyMasked := false
	for row := range p.rows {
		if !p.keyObserved[row*p.tokens+token] {
			anyMasked = true
			continue
		}
		target := (row*count + position) * p.innerWidth
		copy(p.attention[target:target+p.innerWidth], p.value[target:target+p.innerWidth])
	}
	if anyMasked {
		p.uniformGroupAverage(scratch, token, position, count)
	}
}

// uniformGroupAverage writes the uniform average of every row's values at
// token into the attention output of each row whose token is masked; token
// is block position position of a block holding count positions per row.
//
// It uses the same weights and the same kernel procedure as the general
// path, so the results match it bitwise in every build. For an all-masked
// row, every score rounds to exactly maskedScore, so SoftmaxRows sees a
// finite maximum and takes its exponential path: each exp(0) is exactly 1,
// their float32 sum is exactly R, and each weight is 1 · (1/R), which equals
// 1/float32(R) exactly. MatMulTransposed computes each output from its own
// length-R inner product alone, so one weight row per head gives the same
// bits as the general path's R-row product. Heads are processed one at a
// time because worker scratch holds one head's transposed values.
func (p *forwardPass) uniformGroupAverage(scratch *workerScratch, token, position, count int) {
	rows, headWidth := p.rows, p.config.HeadWidth
	weights := scratch.scores[:rows]
	inverseSum := 1 / float32(rows)
	for i := range weights {
		weights[i] = inverseSum
	}
	valueTransposed := scratch.valueTransposed[:rows*headWidth]
	average := scratch.output[:headWidth]
	for head := range p.config.Heads {
		p.gatherHeadTransposed(valueTransposed, p.value, position, count, rows, rows, head)
		kernel.MatMulTransposed(average, weights, valueTransposed, 1, rows, headWidth)
		for row := range rows {
			if p.keyObserved[row*p.tokens+token] {
				continue
			}
			offset := (row*count+position)*p.innerWidth + head*headWidth
			copy(p.attention[offset:offset+headWidth], average)
		}
	}
}
