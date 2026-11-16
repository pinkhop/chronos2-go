package network

import (
	"github.com/pinkhop/chronos2-go/internal/kernel"
	"github.com/pinkhop/chronos2-go/internal/parallel"
)

// feedForward applies the pre-norm feed-forward sublayer, W_out·ReLU(W_in·x),
// and adds it to the residual stream, one block of tokens at a time so the
// hidden activations never need more than blockCapacity × FeedForwardWidth
// values.
func (p *forwardPass) feedForward(weights feedForwardWeights) {
	width, hiddenWidth := p.config.ModelWidth, p.config.FeedForwardWidth
	for first := 0; first < p.totalTokens; first += p.blockCapacity {
		tokens := min(p.blockCapacity, p.totalTokens-first)
		residual := p.hidden[first*width : (first+tokens)*width]
		normed := p.blockInput[:tokens*width]
		activations := p.activations[:tokens*hiddenWidth]
		p.rmsNorm(normed, residual, weights.norm, tokens)
		p.project(projection{out: activations, in: normed, weight: weights.input, relu: true, rows: tokens, inWidth: width, outWidth: hiddenWidth})
		p.project(projection{out: normed, in: activations, weight: weights.output, rows: tokens, inWidth: hiddenWidth, outWidth: width})
		p.addValues(residual, normed)
	}
}

// applyResidualBlock applies block to rows inputs ([rows × inputWidth]) one
// block of rows at a time, computing W_out·ReLU(W_hidden·x + b_hidden) +
// b_out, then adding W_res·x + b_res, in the reference's operation order.
// For each block it calls emit with the index of the block's first row, its
// row count, and its outputs ([count × outputWidth]), which are valid only
// during the call.
func (p *forwardPass) applyResidualBlock(in []float32, rows int, block residualBlockWeights, emit func(firstRow, count int, values []float32)) {
	inputWidth, hiddenWidth, outputWidth := block.inputWidth, block.hiddenWidth, block.outputWidth
	for first := 0; first < rows; first += p.blockCapacity {
		count := min(p.blockCapacity, rows-first)
		input := in[first*inputWidth : (first+count)*inputWidth]
		activations := p.activations[:count*hiddenWidth]
		output := p.residual[:count*outputWidth]
		skip := p.residual[count*outputWidth : 2*count*outputWidth]
		p.project(projection{
			out: activations, in: input, weight: block.hiddenWeight, bias: block.hiddenBias, relu: true,
			rows: count, inWidth: inputWidth, outWidth: hiddenWidth,
		})
		p.project(projection{
			out: output, in: activations, weight: block.outputWeight, bias: block.outputBias,
			rows: count, inWidth: hiddenWidth, outWidth: outputWidth,
		})
		p.project(projection{
			out: skip, in: input, weight: block.residualWeight, bias: block.residualBias,
			rows: count, inWidth: inputWidth, outWidth: outputWidth,
		})
		p.addValues(output, skip)
		emit(first, count, output)
	}
}

// addValues adds source to destination element-wise, in parallel; both have
// the same length.
func (p *forwardPass) addValues(destination, source []float32) {
	parallel.For(len(destination), p.workers, func(start, end int) {
		kernel.Add(destination[start:end], destination[start:end], source[start:end])
	})
}

// finalNormAll applies the encoder's final norm to every token, observes the
// result as the hidden stage, and returns it along with the gathered
// output-patch tokens ([R·P × ModelWidth]) that decode reads.
func (p *forwardPass) finalNormAll() (hidden, forecastTokens []float32) {
	hidden = make([]float32, p.totalTokens*p.config.ModelWidth)
	p.rmsNorm(hidden, p.hidden, p.network.finalNorm, p.totalTokens)
	p.observe("hidden", hidden)
	return hidden, p.gatherForecastTokens(hidden)
}

// finalNormForecast applies the encoder's final norm to just the output-patch
// tokens, the only ones a forecast reads, and returns them as
// [R·P × ModelWidth].
func (p *forwardPass) finalNormForecast() []float32 {
	forecastTokens := p.gatherForecastTokens(p.hidden)
	p.rmsNorm(forecastTokens, forecastTokens, p.network.finalNorm, p.rows*p.outputPatches)
	return forecastTokens
}

// gatherForecastTokens copies each row's output-patch tokens out of states
// ([R × S × ModelWidth]) into a new [R·P × ModelWidth] slice.
func (p *forwardPass) gatherForecastTokens(states []float32) []float32 {
	width := p.config.ModelWidth
	gathered := make([]float32, p.rows*p.outputPatches*width)
	for row := range p.rows {
		first := (row*p.tokens + p.futureOffset) * width
		copy(gathered[row*p.outputPatches*width:], states[first:first+p.outputPatches*width])
	}
	return gathered
}
