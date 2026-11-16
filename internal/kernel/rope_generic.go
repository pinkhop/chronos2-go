package kernel

// ApplyRoPE applies rotary position embeddings to x in place using the
// "rotate half" convention: x·cos + rotateHalf(x)·sin, where
// rotateHalf(x) = concat(−x[headDim/2:], x[:headDim/2]). Element i of a head
// therefore pairs with element i + headDim/2, not with its neighbor.
//
// x is [tokens × heads × headDim], the layout of a projected query or key
// before heads are split out. cos and sin are [tokens × headDim] tables whose
// two halves repeat the same frequencies, as the reference builds them; every
// head of a token shares that token's row. headDim must be even. It panics
// when headDim is odd or a length disagrees with the dimensions.
func ApplyRoPE(x, cos, sin []float32, tokens, heads, headDim int) {
	clearUpperBits()
	if headDim%2 != 0 {
		panic("kernel: ApplyRoPE requires an even headDim")
	}
	requireLength("x", len(x), tokens*heads*headDim)
	requireLength("cos", len(cos), tokens*headDim)
	requireLength("sin", len(sin), tokens*headDim)
	half := headDim / 2
	for token := range tokens {
		cosRow := cos[token*headDim : (token+1)*headDim]
		sinRow := sin[token*headDim : (token+1)*headDim]
		for head := range heads {
			start := (token*heads + head) * headDim
			rotateHalf(x[start:start+headDim], cosRow, sinRow, half)
		}
	}
}

// rotateHalf rotates one head vector in place; vector, cosRow, and sinRow all
// hold 2·half values.
func rotateHalf(vector, cosRow, sinRow []float32, half int) {
	lower, upper := vector[:half], vector[half:2*half]
	cosLower, cosUpper := cosRow[:half], cosRow[half:2*half]
	sinLower, sinUpper := sinRow[:half], sinRow[half:2*half]
	for i := range lower {
		first, second := lower[i], upper[i]
		lower[i] = first*cosLower[i] - second*sinLower[i]
		upper[i] = second*cosUpper[i] + first*sinUpper[i]
	}
}
