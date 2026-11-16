package network

import "math"

// decode turns the final hidden states of each row's output-patch tokens
// (forecastTokens, [R·P × ModelWidth]) into quantile forecasts. It applies the output residual block, rearranges
// each patch's quantile-major values into a [Q × 16P] grid per row, and maps
// the grid back to the caller's units through sinh (when configured), the
// row's scale, and its location. It returns [R × Q × 16P] values.
func (p *forwardPass) decode(forecastTokens []float32) []float32 {
	size := p.config.PatchSize
	quantiles := len(p.config.Quantiles)

	// Decode every output patch.
	patchValues := quantiles * size
	decoded := make([]float32, p.rows*p.outputPatches*patchValues)
	keep := func(firstRow, count int, values []float32) {
		copy(decoded[firstRow*patchValues:(firstRow+count)*patchValues], values)
	}
	p.applyResidualBlock(forecastTokens, p.rows*p.outputPatches, p.network.outputBlock, keep)

	// Rearrange "b n (q p) -> b q (n p)".
	steps := p.outputPatches * size
	normalized := make([]float32, p.rows*quantiles*steps)
	for row := range p.rows {
		for patch := range p.outputPatches {
			source := decoded[(row*p.outputPatches+patch)*patchValues:]
			for level := range quantiles {
				destination := normalized[(row*quantiles+level)*steps+patch*size:]
				copy(destination[:size], source[level*size:(level+1)*size])
			}
		}
	}
	p.observe("quantiles_normalized", normalized)

	// Invert the normalization.
	values := make([]float32, len(normalized))
	perRow := quantiles * steps
	for i, value := range normalized {
		if p.config.UseArcsinh {
			value = float32(math.Sinh(float64(value)))
		}
		row := i / perRow
		values[i] = value*p.scales[row] + p.locations[row]
	}
	p.observe("quantiles", values)
	return values
}
