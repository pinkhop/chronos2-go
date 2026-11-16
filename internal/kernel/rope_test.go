package kernel

import (
	"math"
	"testing"
)

// ropeTables builds rotate-half cos and sin tables of shape
// [tokens × headDim] the way the reference does: per-pair inverse
// frequencies θ^(−2i/headDim), concatenated with themselves.
func ropeTables(tokens, headDim int, theta float64) (cos, sin []float32) {
	cos = make([]float32, tokens*headDim)
	sin = make([]float32, tokens*headDim)
	half := headDim / 2
	for position := range tokens {
		for i := range half {
			angle := float64(position) / math.Pow(theta, float64(2*i)/float64(headDim))
			for _, column := range []int{i, i + half} {
				cos[position*headDim+column] = float32(math.Cos(angle))
				sin[position*headDim+column] = float32(math.Sin(angle))
			}
		}
	}
	return cos, sin
}

// referenceApplyRoPE computes x·cos + rotateHalf(x)·sin in float64, where
// rotateHalf(x) = concat(−x[half:], x[:half]) and x is
// [tokens × heads × headDim].
func referenceApplyRoPE(x, cos, sin []float32, tokens, heads, headDim int) []float64 {
	out := make([]float64, len(x))
	half := headDim / 2
	for token := range tokens {
		for head := range heads {
			base := (token*heads + head) * headDim
			for d := range headDim {
				var rotated float64
				if d < half {
					rotated = -float64(x[base+d+half])
				} else {
					rotated = float64(x[base+d-half])
				}
				table := token*headDim + d
				out[base+d] = float64(x[base+d])*float64(cos[table]) + rotated*float64(sin[table])
			}
		}
	}
	return out
}

// TestApplyRoPE_MultipleTokensAndHeads_MatchesReference verifies that ApplyRoPE, given multiple tokens and heads, matches reference.
func TestApplyRoPE_MultipleTokensAndHeads_MatchesReference(t *testing.T) {
	t.Parallel()

	// Given
	const tokens, heads, headDim = 5, 3, 8
	x := randomValues(10, tokens*heads*headDim)
	cos, sin := ropeTables(tokens, headDim, 10000)
	want := referenceApplyRoPE(x, cos, sin, tokens, heads, headDim)

	// When
	ApplyRoPE(x, cos, sin, tokens, heads, headDim)

	// Then
	assertClose(t, "x", x, want)
}

// TestApplyRoPE_QuarterTurn_RotatesHalvesAsPairs verifies that ApplyRoPE, given quarter turn, rotates halves as pairs.
func TestApplyRoPE_QuarterTurn_RotatesHalvesAsPairs(t *testing.T) {
	t.Parallel()

	// Given: cos 0 and sin 1 everywhere, a quarter turn of every pair.
	x := []float32{1, 2, 3, 4}
	cos := []float32{0, 0, 0, 0}
	sin := []float32{1, 1, 1, 1}

	// When
	ApplyRoPE(x, cos, sin, 1, 1, 4)

	// Then: pairs are (x[0], x[2]) and (x[1], x[3]), not adjacent elements.
	assertClose(t, "x", x, []float64{-3, -4, 1, 2})
}

// TestApplyRoPE_PositionZero_LeavesValuesUnchanged verifies that ApplyRoPE, given position zero, leaves values unchanged.
func TestApplyRoPE_PositionZero_LeavesValuesUnchanged(t *testing.T) {
	t.Parallel()

	// Given
	const heads, headDim = 2, 64
	x := randomValues(11, heads*headDim)
	want := make([]float64, len(x))
	for i, v := range x {
		want[i] = float64(v)
	}
	cos, sin := ropeTables(1, headDim, 10000)

	// When
	ApplyRoPE(x, cos, sin, 1, heads, headDim)

	// Then
	assertClose(t, "x", x, want)
}

// TestApplyRoPE_OddHeadDim_Panics verifies that ApplyRoPE, given odd head dim, panics.
func TestApplyRoPE_OddHeadDim_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 3)
	cos := make([]float32, 3)
	sin := make([]float32, 3)

	// When / Then
	assertPanics(t, func() { ApplyRoPE(x, cos, sin, 1, 1, 3) })
}

// TestApplyRoPE_TableLengthMismatch_Panics verifies that ApplyRoPE, given table length mismatch, panics.
func TestApplyRoPE_TableLengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 8)
	cos := make([]float32, 4)
	sin := make([]float32, 3)

	// When / Then
	assertPanics(t, func() { ApplyRoPE(x, cos, sin, 1, 2, 4) })
}
