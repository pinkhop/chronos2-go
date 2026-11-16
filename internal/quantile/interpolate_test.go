package quantile_test

import (
	"testing"

	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// TestInterpolate_InteriorLevels_PadsBoundariesAndInterpolates verifies that Interpolate, given interior levels, pads boundaries and interpolates.
func TestInterpolate_InteriorLevels_PadsBoundariesAndInterpolates(t *testing.T) {
	t.Parallel()

	// Given three levels with two steps each; padding repeats the first line
	// at level 0 and the last line at level 1.
	levels := []float64{0.1, 0.5, 0.9}
	values := [][]float64{{1, 10}, {2, 20}, {4, 40}}
	queries := []float64{0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.95, 1}

	// When
	got := quantile.Interpolate(queries, levels, values)

	// Then
	want := [][]float64{{1, 10}, {1, 10}, {1, 10}, {1.5, 15}, {2, 20}, {3, 30}, {4, 40}, {4, 40}}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		assertClose(t, "query "+formatLevel(queries[i]), got[i], want[i])
	}
}

// TestInterpolate_LevelsIncludeZeroAndOne_NoPadding verifies that Interpolate adds no padding when the levels include zero and one.
func TestInterpolate_LevelsIncludeZeroAndOne_NoPadding(t *testing.T) {
	t.Parallel()

	// Given
	levels := []float64{0, 0.5, 1}
	values := [][]float32{{0}, {5}, {10}}

	// When
	got := quantile.Interpolate([]float64{0, 0.25, 0.75, 1}, levels, values)

	// Then
	assertClose(t, "column", column(got), []float32{0, 2.5, 7.5, 10})
}

// TestInterpolate_UnsortedLevels_SortsWithValues verifies that Interpolate, given unsorted levels, sorts with values.
func TestInterpolate_UnsortedLevels_SortsWithValues(t *testing.T) {
	t.Parallel()

	// Given
	levels := []float64{0.9, 0.1}
	values := [][]float64{{9}, {1}}

	// When
	got := quantile.Interpolate([]float64{0.05, 0.5, 0.95}, levels, values)

	// Then
	assertClose(t, "column", column(got), []float64{1, 5, 9})
}

// TestInterpolate_TiedTopLevelsAtQueryOne_TakesLowerValue verifies that Interpolate, given tied top levels at query one, takes lower value.
func TestInterpolate_TiedTopLevelsAtQueryOne_TakesLowerValue(t *testing.T) {
	t.Parallel()

	// Given two lines at level 1. The reference clamps the bracket to the last
	// two levels, divides 0 by 0, and maps the NaN weight to 0.
	levels := []float64{0.5, 1, 1}
	values := [][]float64{{1}, {2}, {3}}

	// When
	got := quantile.Interpolate([]float64{1}, levels, values)

	// Then
	assertClose(t, "column", column(got), []float64{2})
}

// TestInterpolate_Float64Values_ComputesInFloat32 verifies that Interpolate computes in float32 for float64 values.
func TestInterpolate_Float64Values_ComputesInFloat32(t *testing.T) {
	t.Parallel()

	// Given a value float32 cannot represent exactly.
	levels := []float64{0.5}
	values := [][]float64{{0.1}}

	// When
	got := quantile.Interpolate([]float64{0.5}, levels, values)

	// Then
	if want := float64(float32(0.1)); got[0][0] != want {
		t.Errorf("got %v, want %v (0.1 rounded through float32)", got[0][0], want)
	}
}

// TestInterpolate_InvalidArguments_Panics verifies that Interpolate, given invalid arguments, panics.
func TestInterpolate_InvalidArguments_Panics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		queries []float64
		levels  []float64
		values  [][]float64
	}{
		{"no levels", []float64{0.5}, nil, nil},
		{"levels and values differ", []float64{0.5}, []float64{0.5}, [][]float64{{1}, {2}}},
		{"ragged lines", []float64{0.5}, []float64{0.1, 0.9}, [][]float64{{1}, {2, 3}}},
		{"query above one", []float64{1.5}, []float64{0.5}, [][]float64{{1}}},
		{"query NaN", []float64{nan()}, []float64{0.5}, [][]float64{{1}}},
		{"level below zero", []float64{0.5}, []float64{-0.1}, [][]float64{{1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt

			// When
			recovered := panicValue(func() { quantile.Interpolate(tt.queries, tt.levels, tt.values) })

			// Then
			if recovered == nil {
				t.Errorf("Interpolate() did not panic")
			}
		})
	}
}
