package network

import (
	"github.com/pinkhop/chronos2-go/internal/kernel"
	"github.com/pinkhop/chronos2-go/internal/parallel"
)

// timeAttention applies the pre-norm time self-attention sublayer: each row
// attends across its own tokens, with rotary embeddings on queries and keys
// and the row's patch mask applied additively to keys. Rows are processed in
// blocks whose tokens are contiguous in the residual stream.
func (p *forwardPass) timeAttention(weights attentionWeights) {
	width, heads := p.config.ModelWidth, p.config.Heads
	queryBlocks := (p.tokens + queryBlock - 1) / queryBlock
	for firstRow := 0; firstRow < p.rows; firstRow += p.timeBlockRows {
		rows := min(p.timeBlockRows, p.rows-firstRow)
		tokens := rows * p.tokens
		residual := p.hidden[firstRow*p.tokens*width : (firstRow+rows)*p.tokens*width]
		p.projectHeads(weights, residual, tokens)
		p.rotate(rows)
		p.forEachUnit(rows*heads, func(_ *workerScratch, unit int) {
			p.layoutHead(unit/heads, unit%heads)
		})
		p.forEachUnit(rows*heads*queryBlocks, func(scratch *workerScratch, unit int) {
			row, head, block := unit/(heads*queryBlocks), unit/queryBlocks%heads, unit%queryBlocks
			p.timeAttentionUnit(scratch, firstRow, row, head, block)
		})
		p.finishAttention(weights, residual, tokens)
	}
}

// rotate applies rotary embeddings to the block's queries and keys, row by
// row, with positions counted from each row's first token.
func (p *forwardPass) rotate(rows int) {
	rowValues := p.tokens * p.innerWidth
	parallel.For(rows, p.workers, func(start, end int) {
		for row := start; row < end; row++ {
			kernel.ApplyRoPE(p.query[row*rowValues:(row+1)*rowValues], p.cos, p.sin, p.tokens, p.config.Heads, p.config.HeadWidth)
			kernel.ApplyRoPE(p.key[row*rowValues:(row+1)*rowValues], p.cos, p.sin, p.tokens, p.config.Heads, p.config.HeadWidth)
		}
	})
}

// headSpan returns the extent of one (row, head) region of headKey and
// headValue, where row is the row's index within the current time-attention
// block.
func (p *forwardPass) headSpan(row, head int) (start, end int) {
	size := p.paddedTokens * p.config.HeadWidth
	start = (row*p.config.Heads + head) * size
	return start, start + size
}

// layoutHead copies one head of one block row's keys into headKey and its
// values, transposed, into headValue, once for all of the row's query
// blocks. It writes only real tokens, leaving the zero padding in place.
func (p *forwardPass) layoutHead(row, head int) {
	start, end := p.headSpan(row, head)
	first := row * p.tokens
	p.gatherHead(p.headKey[start:end], p.key, first, 1, p.tokens, head)
	p.gatherHeadTransposed(p.headValue[start:end], p.value, first, 1, p.tokens, p.paddedTokens, head)
}

// timeAttentionUnit computes one head's attention for one block of queries
// of one row: row is the row's index within the block starting at firstRow,
// and block selects queries [block·queryBlock, (block+1)·queryBlock). It
// attends over the row's paddedTokens keys in headKey and headValue, where
// the padding keys' negative-infinity mask gives them zero weight.
func (p *forwardPass) timeAttentionUnit(scratch *workerScratch, firstRow, row, head, block int) {
	keys, headWidth := p.paddedTokens, p.config.HeadWidth
	first := row * p.tokens
	firstQuery := block * queryBlock
	queries := min(queryBlock, p.tokens-firstQuery)
	start, end := p.headSpan(row, head)
	key, valueTransposed := p.headKey[start:end], p.headValue[start:end]

	// Gather this block's queries.
	query := scratch.query[:queries*headWidth]
	p.gatherHead(query, p.query, first+firstQuery, 1, queries, head)

	// Scores are deliberately unscaled: the checkpoint was trained without
	// the usual 1/√d factor.
	scores := scratch.scores[:queries*keys]
	kernel.MatMulTransposed(scores, query, key, queries, headWidth, keys)
	maskRow := (firstRow + row) * keys
	kernel.AddBias(scores, p.timeMask[maskRow:maskRow+keys], queries, keys)
	kernel.SoftmaxRows(scores, queries, keys)
	output := scratch.output[:queries*headWidth]
	kernel.MatMulTransposed(output, scores, valueTransposed, queries, keys, headWidth)
	p.scatterHead(output, first+firstQuery, 1, queries, head)
}

// projectHeads normalizes tokens rows of input into blockInput and projects
// them to queries, keys, and values.
func (p *forwardPass) projectHeads(weights attentionWeights, input []float32, tokens int) {
	width := p.config.ModelWidth
	normed := p.blockInput[:tokens*width]
	p.rmsNorm(normed, input, weights.norm, tokens)
	for _, target := range []struct{ out, weight []float32 }{
		{p.query, weights.query}, {p.key, weights.key}, {p.value, weights.value},
	} {
		p.project(projection{
			out: target.out[:tokens*p.innerWidth], in: normed, weight: target.weight,
			rows: tokens, inWidth: width, outWidth: p.innerWidth,
		})
	}
}

// finishAttention projects the block's concatenated heads back to the model
// width and adds the result to residual, the block's slice of the residual
// stream (or, for group attention, its gathered copy).
func (p *forwardPass) finishAttention(weights attentionWeights, residual []float32, tokens int) {
	width := p.config.ModelWidth
	output := p.blockInput[:tokens*width]
	p.project(projection{
		out: output, in: p.attention[:tokens*p.innerWidth], weight: weights.output,
		rows: tokens, inWidth: p.innerWidth, outWidth: width,
	})
	p.addValues(residual, output)
}

// gatherHead copies head's slice of count block tokens from source
// ([block tokens × innerWidth]) into destination ([count × HeadWidth]);
// member i is block token first + i·stride.
func (p *forwardPass) gatherHead(destination, source []float32, first, stride, count, head int) {
	headWidth := p.config.HeadWidth
	for member := range count {
		offset := (first+member*stride)*p.innerWidth + head*headWidth
		copy(destination[member*headWidth:(member+1)*headWidth], source[offset:offset+headWidth])
	}
}

// gatherHeadTransposed is gatherHead with the destination transposed to
// [HeadWidth × columns], so attention weights can multiply values through
// the faster transposed matrix product; columns ≥ count is the destination's
// row stride, and columns past count are left untouched.
func (p *forwardPass) gatherHeadTransposed(destination, source []float32, first, stride, count, columns, head int) {
	headWidth := p.config.HeadWidth
	for member := range count {
		offset := (first+member*stride)*p.innerWidth + head*headWidth
		for d, value := range source[offset : offset+headWidth] {
			destination[d*columns+member] = value
		}
	}
}

// scatterHead copies source ([count × HeadWidth]) into head's slice of the
// attention buffer, the inverse of gatherHead.
func (p *forwardPass) scatterHead(source []float32, first, stride, count, head int) {
	headWidth := p.config.HeadWidth
	for member := range count {
		offset := (first+member*stride)*p.innerWidth + head*headWidth
		copy(p.attention[offset:offset+headWidth], source[member*headWidth:(member+1)*headWidth])
	}
}
