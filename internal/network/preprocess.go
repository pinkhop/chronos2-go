package network

import (
	"math"
)

// minimumScale replaces a zero standard deviation, as the reference's
// InstanceNorm does, so constant rows normalize to zero instead of NaN.
const minimumScale = 1e-5

// maskedScore is the additive attention mask for a masked key: the most
// negative finite float32, as the reference uses (torch.finfo.min).
const maskedScore = -math.MaxFloat32

// paddingScore is the additive time-attention mask for a padding key, which
// the reference does not have. Negative infinity gives the key exactly zero
// weight; maskedScore would give it the same weight as masked real keys in a
// row whose real keys are all masked.
var paddingScore = float32(math.Inf(-1))

// normalize computes each row's location and scale over its observed values,
// standardizes the context (then applies asinh when configured), and returns
// the normalized [R × L] context with NaN wherever the input is missing.
func (p *forwardPass) normalize() []float32 {
	length := p.batch.ContextLength
	normalized := make([]float32, p.rows*length)
	for row := range p.rows {
		values := p.batch.Context[row*length : (row+1)*length]
		location, scale := rowStatistics(values)
		p.locations[row], p.scales[row] = location, scale
		for i, value := range values {
			normalized[row*length+i] = p.transform(value, location, scale)
		}
	}
	p.observe("loc", p.locations)
	p.observe("scale", p.scales)
	p.observe("normalized_context", normalized)
	return normalized
}

// rowStatistics returns the mean and population standard deviation of the
// row's non-NaN values. A row with no observations gets location 0 and scale
// 1; a zero standard deviation becomes minimumScale. Sums accumulate in
// float64 and round once, so a constant row gets its exact value as location
// and minimumScale as scale. The reference's float32 sums can land one ulp off
// on such a row; chronos2-go deliberately does not reproduce that rounding
// (see "How Chronos-2 Works" in docs/DESIGN.md).
func rowStatistics(values []float32) (location, scale float32) {
	var sum float64
	count := 0
	for _, value := range values {
		if !isNaN(value) {
			sum += float64(value)
			count++
		}
	}
	if count == 0 {
		return 0, 1
	}
	location = float32(sum / float64(count))
	var squares float64
	for _, value := range values {
		if !isNaN(value) {
			deviation := float64(value - location)
			squares += deviation * deviation
		}
	}
	scale = float32(math.Sqrt(squares / float64(count)))
	if scale == 0 {
		scale = minimumScale
	}
	return location, scale
}

// transform standardizes value by a row's location and scale and applies
// asinh when the model uses it; NaN stays NaN.
func (p *forwardPass) transform(value, location, scale float32) float32 {
	standardized := (value - location) / scale
	if p.config.UseArcsinh {
		return float32(math.Asinh(float64(standardized)))
	}
	return standardized
}

// isNaN reports whether value is NaN.
func isNaN(value float32) bool {
	return value != value
}

// prepareMasks fills the per-token attention masks and the group matrix. A
// context patch may be attended to when it holds at least one observation;
// the [REG] token and every output patch always may.
func (p *forwardPass) prepareMasks() {
	size, length := p.config.PatchSize, p.batch.ContextLength
	padding := p.contextPatches*size - length
	for row := range p.rows {
		for token := range p.tokens {
			observed := token >= p.contextPatches
			for step := range size {
				if observed {
					break
				}
				index := token*size + step - padding
				observed = index >= 0 && !isNaN(p.batch.Context[row*length+index])
			}
			p.keyObserved[row*p.tokens+token] = observed
			if !observed {
				p.timeMask[row*p.paddedTokens+token] = maskedScore
			}
		}
		for token := p.tokens; token < p.paddedTokens; token++ {
			p.timeMask[row*p.paddedTokens+token] = paddingScore
		}
		for other := range p.rows {
			p.sameGroup[row*p.rows+other] = p.batch.GroupIDs[row] == p.batch.GroupIDs[other]
		}
	}
}

// embed builds the context and future patch features, embeds them with the
// input residual block, and assembles the residual stream: context tokens,
// the [REG] token, then output-patch tokens for each row.
func (p *forwardPass) embed(normalized []float32) {
	width := p.config.ModelWidth

	// Embed context patches into each row's leading tokens.
	patchedContext := p.patchContext(normalized)
	p.observe("patched_context", patchedContext)
	p.applyResidualBlock(patchedContext, p.rows*p.contextPatches, p.network.inputBlock, func(firstRow, count int, values []float32) {
		p.placeTokens(values, firstRow, count, p.contextPatches, 0)
	})

	// Insert the [REG] token after each row's context tokens.
	if p.network.regEmbedding != nil {
		for row := range p.rows {
			copy(p.hidden[(row*p.tokens+p.contextPatches)*width:], p.network.regEmbedding)
		}
	}

	// Embed future patches into each row's trailing tokens.
	patchedFuture := p.patchFuture()
	p.observe("patched_future", patchedFuture)
	p.applyResidualBlock(patchedFuture, p.rows*p.outputPatches, p.network.inputBlock, func(firstRow, count int, values []float32) {
		p.placeTokens(values, firstRow, count, p.outputPatches, p.futureOffset)
	})
	p.observe("input_embeds", p.hidden)
}

// placeTokens copies count embedded patches, starting at flat patch index
// firstPatch of a [R × perRow] patch grid, into the residual stream at token
// offset + (index within the row) of each patch's row.
func (p *forwardPass) placeTokens(values []float32, firstPatch, count, perRow, offset int) {
	width := p.config.ModelWidth
	for i := range count {
		patch := firstPatch + i
		token := patch/perRow*p.tokens + offset + patch%perRow
		copy(p.hidden[token*width:(token+1)*width], values[i*width:(i+1)*width])
	}
}

// patchContext left-pads each normalized row with missing steps to whole
// patches and returns [R × N × 3·PatchSize] features: per patch, the time
// encoding, the values (0 where missing), and the observation mask.
func (p *forwardPass) patchContext(normalized []float32) []float32 {
	size, length := p.config.PatchSize, p.batch.ContextLength
	features := size * patchFeatures
	padding := p.contextPatches*size - length
	patched := make([]float32, p.rows*p.contextPatches*features)
	firstOffset := -p.contextPatches * size
	for row := range p.rows {
		for patch := range p.contextPatches {
			out := patched[(row*p.contextPatches+patch)*features : (row*p.contextPatches+patch+1)*features]
			for step := range size {
				out[step] = p.timeEncoding(firstOffset + patch*size + step)
				index := patch*size + step - padding
				if index >= 0 && !isNaN(p.batch.Context[row*length+index]) {
					out[size+step] = normalized[row*length+index]
					out[2*size+step] = 1
				}
			}
		}
	}
	return patched
}

// patchFuture returns [R × P × 3·PatchSize] features for the output patches:
// the time encoding, each known future covariate value transformed by its
// row's statistics (0 where unknown), and the mask of known values.
func (p *forwardPass) patchFuture() []float32 {
	size := p.config.PatchSize
	features := size * patchFeatures
	futureLength := p.outputPatches * size
	patched := make([]float32, p.rows*p.outputPatches*features)
	for row := range p.rows {
		for patch := range p.outputPatches {
			out := patched[(row*p.outputPatches+patch)*features : (row*p.outputPatches+patch+1)*features]
			for step := range size {
				out[step] = p.timeEncoding(patch*size + step)
				value := p.transform(p.batch.FutureCovariates[row*futureLength+patch*size+step], p.locations[row], p.scales[row])
				if !isNaN(value) {
					out[size+step] = value
					out[2*size+step] = 1
				}
			}
		}
	}
	return patched
}

// timeEncoding returns a step offset divided by the time-encoding scale in
// float32, as the reference computes it.
func (p *forwardPass) timeEncoding(offset int) float32 {
	return float32(offset) / float32(p.config.TimeEncodingScale)
}

// rotaryTables returns rotate-half cos and sin tables of shape
// [tokens × headWidth] for positions 0 through tokens−1, reproducing the
// reference's float32 inverse frequencies θ^(−2i/headWidth) with both halves
// of each row repeating the same angles.
func rotaryTables(tokens, headWidth int, theta float64) (cos, sin []float32) {
	half := headWidth / 2
	inverseFrequencies := make([]float32, half)
	for i := range inverseFrequencies {
		exponent := float32(2*i) / float32(headWidth)
		inverseFrequencies[i] = 1 / float32(math.Pow(theta, float64(exponent)))
	}
	cos = make([]float32, tokens*headWidth)
	sin = make([]float32, tokens*headWidth)
	for position := range tokens {
		row := position * headWidth
		for i, frequency := range inverseFrequencies {
			angle := float64(frequency * float32(position))
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			cos[row+i], cos[row+half+i] = c, c
			sin[row+i], sin[row+half+i] = s, s
		}
	}
	return cos, sin
}
