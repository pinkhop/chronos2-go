package kernel

import (
	"math"
	"slices"
	"strconv"
	"testing"
)

// TestAddBias_TwoRows_AddsBiasToEachRow verifies that AddBias, given two rows, adds bias to each row.
func TestAddBias_TwoRows_AddsBiasToEachRow(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{1, 2, 3, 4, 5, 6}
	bias := []float32{10, 20, 30}

	// When
	AddBias(x, bias, 2, 3)

	// Then
	if want := []float32{11, 22, 33, 14, 25, 36}; !slices.Equal(x, want) {
		t.Errorf("x = %v, want %v", x, want)
	}
}

// TestAddBias_BiasLengthMismatch_Panics verifies that AddBias, given bias length mismatch, panics.
func TestAddBias_BiasLengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	x := make([]float32, 6)
	bias := make([]float32, 2)

	// When / Then
	assertPanics(t, func() { AddBias(x, bias, 2, 3) })
}

// TestReLU_MixedSigns_ClampsNegativesToZero verifies that ReLU, given mixed signs, clamps negatives to zero.
func TestReLU_MixedSigns_ClampsNegativesToZero(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{-2, -0.5, 0, 0.5, 2, -1, 3}

	// When
	ReLU(x)

	// Then
	if want := []float32{0, 0, 0, 0.5, 2, 0, 3}; !slices.Equal(x, want) {
		t.Errorf("x = %v, want %v", x, want)
	}
}

// TestReLU_NaN_StaysNaN verifies that ReLU, given NaN, stays NaN.
func TestReLU_NaN_StaysNaN(t *testing.T) {
	t.Parallel()

	// Given
	x := []float32{float32(math.NaN())}

	// When
	ReLU(x)

	// Then
	if !math.IsNaN(float64(x[0])) {
		t.Errorf("x[0] = %g, want NaN", x[0])
	}
}

// TestAdd_DistinctOutput_WritesElementwiseSum verifies that Add, given distinct output, writes elementwise sum.
func TestAdd_DistinctOutput_WritesElementwiseSum(t *testing.T) {
	t.Parallel()

	// Given
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{10, 20, 30, 40, 50}
	out := make([]float32, 5)

	// When
	Add(out, a, b)

	// Then
	if want := []float32{11, 22, 33, 44, 55}; !slices.Equal(out, want) {
		t.Errorf("out = %v, want %v", out, want)
	}
}

// TestAdd_OutputAliasesInput_AccumulatesInPlace verifies that Add, given output aliases input, accumulates in place.
func TestAdd_OutputAliasesInput_AccumulatesInPlace(t *testing.T) {
	t.Parallel()

	// Given
	a := []float32{1, 2, 3}
	b := []float32{10, 20, 30}

	// When
	Add(a, a, b)

	// Then
	if want := []float32{11, 22, 33}; !slices.Equal(a, want) {
		t.Errorf("a = %v, want %v", a, want)
	}
}

// TestAdd_LengthMismatch_Panics verifies that Add, given length mismatch, panics.
func TestAdd_LengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	a := make([]float32, 3)
	b := make([]float32, 4)
	out := make([]float32, 3)

	// When / Then
	assertPanics(t, func() { Add(out, a, b) })
}

// TestDot_VariousLengths_MatchesReference verifies that Dot, given various lengths, matches reference.
func TestDot_VariousLengths_MatchesReference(t *testing.T) {
	t.Parallel()
	for _, length := range []int{0, 1, 3, 4, 7, 64, 769} {
		t.Run("length "+strconv.Itoa(length), func(t *testing.T) {
			t.Parallel()

			// Given
			a := randomValues(5, length)
			b := randomValues(6, length)
			var want float64
			for i := range length {
				want += float64(a[i]) * float64(b[i])
			}

			// When
			got := Dot(a, b)

			// Then
			assertClose(t, "dot", []float32{got}, []float64{want})
		})
	}
}

// TestDot_LengthMismatch_Panics verifies that Dot, given length mismatch, panics.
func TestDot_LengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	a := make([]float32, 3)
	b := make([]float32, 2)

	// When / Then
	assertPanics(t, func() { Dot(a, b) })
}
