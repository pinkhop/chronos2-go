package kernel

import (
	"math"
	"slices"
	"testing"
)

// referenceRMSNorm computes T5-style RMS normalization per row in float64.
func referenceRMSNorm(x, weight []float32, rows, cols int, eps float64) []float64 {
	out := make([]float64, rows*cols)
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		var sumOfSquares float64
		for _, v := range row {
			sumOfSquares += float64(v) * float64(v)
		}
		inverseRMS := 1 / math.Sqrt(sumOfSquares/float64(cols)+eps)
		for c, v := range row {
			out[r*cols+c] = float64(weight[c]) * float64(v) * inverseRMS
		}
	}
	return out
}

// referenceSoftmaxRows computes a numerically stable softmax per row in
// float64.
func referenceSoftmaxRows(x []float32, rows, cols int) []float64 {
	out := make([]float64, rows*cols)
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		maximum := math.Inf(-1)
		for _, v := range row {
			maximum = math.Max(maximum, float64(v))
		}
		var sum float64
		for c, v := range row {
			out[r*cols+c] = math.Exp(float64(v) - maximum)
			sum += out[r*cols+c]
		}
		for c := range row {
			out[r*cols+c] /= sum
		}
	}
	return out
}

// TestRMSNorm_RandomRows_MatchesReference verifies that RMSNorm, given random rows, matches reference.
func TestRMSNorm_RandomRows_MatchesReference(t *testing.T) {
	t.Parallel()

	// Given
	const rows, cols = 3, 77
	x := randomValues(7, rows*cols)
	weight := randomValues(8, cols)
	out := make([]float32, rows*cols)

	// When
	RMSNorm(out, x, weight, rows, cols, 1e-6)

	// Then
	assertClose(t, "out", out, referenceRMSNorm(x, weight, rows, cols, 1e-6))
}

// TestRMSNorm_NonzeroMean_DoesNotSubtractMean verifies that RMSNorm, given nonzero mean, does not subtract mean.
func TestRMSNorm_NonzeroMean_DoesNotSubtractMean(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{3, 3, 3, 3}
	weight := []float32{1, 1, 1, 1}
	out := make([]float32, 4)

	// When
	RMSNorm(out, x, weight, 1, 4, 0)

	// Then
	assertClose(t, "out", out, []float64{1, 1, 1, 1})
}

// TestRMSNorm_ZeroRow_UsesEpsilonAndStaysFinite verifies that RMSNorm, given zero row, uses epsilon and stays finite.
func TestRMSNorm_ZeroRow_UsesEpsilonAndStaysFinite(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{0, 0}
	weight := []float32{1, 1}
	out := []float32{7, 7}

	// When
	RMSNorm(out, x, weight, 1, 2, 1e-6)

	// Then
	assertClose(t, "out", out, []float64{0, 0})
}

// TestRMSNorm_WeightLengthMismatch_Panics verifies that RMSNorm, given weight length mismatch, panics.
func TestRMSNorm_WeightLengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 6)
	weight := make([]float32, 2)
	out := make([]float32, 6)

	// When / Then
	assertPanics(t, func() { RMSNorm(out, x, weight, 2, 3, 1e-6) })
}

// TestSoftmaxRows_RandomRows_MatchesReference verifies that SoftmaxRows, given random rows, matches reference.
func TestSoftmaxRows_RandomRows_MatchesReference(t *testing.T) {
	t.Parallel()

	// Given
	const rows, cols = 4, 37
	x := randomValues(9, rows*cols)
	for i := range x {
		x[i] *= 30
	}
	want := referenceSoftmaxRows(x, rows, cols)

	// When
	SoftmaxRows(x, rows, cols)

	// Then
	assertClose(t, "x", x, want)
}

// TestSoftmaxRows_LargeScores_DoesNotOverflow verifies that SoftmaxRows, given large scores, does not overflow.
func TestSoftmaxRows_LargeScores_DoesNotOverflow(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{1000, 1000, 999}
	want := referenceSoftmaxRows(x, 1, 3)

	// When
	SoftmaxRows(x, 1, 3)

	// Then
	assertClose(t, "x", x, want)
}

// TestSoftmaxRows_AdditivelyMaskedKeys_GetZeroWeight verifies that SoftmaxRows gives additively masked keys zero weight.
func TestSoftmaxRows_AdditivelyMaskedKeys_GetZeroWeight(t *testing.T) {
	t.Parallel()

	// Given
	const masked = -math.MaxFloat32
	x := []float32{2 + masked, 0.5, masked, 0.5}

	// When
	SoftmaxRows(x, 1, 4)

	// Then
	assertClose(t, "x", x, []float64{0, 0.5, 0, 0.5})
}

// TestSoftmaxRows_EveryKeyMasked_IsUniform verifies that SoftmaxRows, given every key masked, is uniform.
func TestSoftmaxRows_EveryKeyMasked_IsUniform(t *testing.T) {
	t.Parallel()

	// Given
	const masked = -math.MaxFloat32
	x := []float32{masked, masked, masked, masked, 1, 1}

	// When
	SoftmaxRows(x, 3, 2)

	// Then
	assertClose(t, "x", x, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5})
}

// TestSoftmaxRows_EveryScoreNegativeInfinity_IsUniformNotNaN verifies that SoftmaxRows, given every score negative infinity, is uniform not NaN.
func TestSoftmaxRows_EveryScoreNegativeInfinity_IsUniformNotNaN(t *testing.T) {
	t.Parallel()

	// Given
	negativeInfinity := float32(math.Inf(-1))
	x := []float32{negativeInfinity, negativeInfinity}

	// When
	SoftmaxRows(x, 1, 2)

	// Then
	assertClose(t, "x", x, []float64{0.5, 0.5})
}

// TestSoftmaxRows_MaskedKeysWithInfinitePadding_IgnoresPadding verifies that
// keys scored negative infinity get exactly zero weight even when every other
// key carries the finite mask, so time attention's padding keys never change
// a fully masked row's uniform weights.
func TestSoftmaxRows_MaskedKeysWithInfinitePadding_IgnoresPadding(t *testing.T) {
	t.Parallel()

	// Given
	const masked = -math.MaxFloat32
	padding := float32(math.Inf(-1))
	x := []float32{masked, masked, masked, padding, padding, padding, padding, padding}

	// When
	SoftmaxRows(x, 1, len(x))

	// Then
	third := float32(1) / 3
	if want := []float32{third, third, third, 0, 0, 0, 0, 0}; !slices.Equal(x, want) {
		t.Errorf("x = %v, want %v", x, want)
	}
}

// TestSoftmaxRows_LengthMismatch_Panics verifies that SoftmaxRows, given length mismatch, panics.
func TestSoftmaxRows_LengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 5)

	// When / Then
	assertPanics(t, func() { SoftmaxRows(x, 2, 3) })
}
