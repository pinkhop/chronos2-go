package kernel

import (
	"math"
	"slices"
	"testing"
)

// TestAddBiasReLU_TwoRows_AddsBiasThenClampsNegatives verifies that
// AddBiasReLU adds bias to each row and then replaces negative sums with
// zero.
func TestAddBiasReLU_TwoRows_AddsBiasThenClampsNegatives(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{1, -2, 3, -4, 5, -6}
	bias := []float32{-3, 1, 0.5}

	// When
	AddBiasReLU(x, bias, 2, 3)

	// Then
	if want := []float32{0, 0, 3.5, 0, 6, 0}; !slices.Equal(x, want) {
		t.Errorf("x = %v, want %v", x, want)
	}
}

// TestAddBiasReLU_RandomValues_MatchesAddBiasThenReLUBitwise verifies that
// the fused kernel gives exactly the bits of AddBias followed by ReLU, so
// replacing the two calls changes no forecast.
func TestAddBiasReLU_RandomValues_MatchesAddBiasThenReLUBitwise(t *testing.T) {
	t.Parallel()

	// Given
	const rows, cols = 7, 37
	want := randomValues(21, rows*cols)
	got := slices.Clone(want)
	bias := randomValues(22, cols)
	AddBias(want, bias, rows, cols)
	ReLU(want)

	// When
	AddBiasReLU(got, bias, rows, cols)

	// Then
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("x[%d] = %g, AddBias then ReLU gave %g", i, got[i], want[i])
		}
	}
}

// TestAddBiasReLU_NaN_StaysNaN verifies that a NaN element or bias stays NaN,
// as in PyTorch.
func TestAddBiasReLU_NaN_StaysNaN(t *testing.T) {
	t.Parallel()

	// Given
	nan := float32(math.NaN())
	x := []float32{nan, -1}
	bias := []float32{1, nan}

	// When
	AddBiasReLU(x, bias, 1, 2)

	// Then
	if !math.IsNaN(float64(x[0])) || !math.IsNaN(float64(x[1])) {
		t.Errorf("x = %v, want [NaN NaN]", x)
	}
}

// TestAddBiasReLU_BiasLengthMismatch_Panics verifies that AddBiasReLU panics
// when bias does not hold cols values.
func TestAddBiasReLU_BiasLengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 6)
	bias := make([]float32, 2)

	// When / Then
	assertPanics(t, func() { AddBiasReLU(x, bias, 2, 3) })
}
