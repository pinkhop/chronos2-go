package quantile_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// TestWeightedQuantile_UnequalWeights_InterpolatesEmpiricalCDF verifies that WeightedQuantile, given unequal weights, interpolates the empirical CDF.
func TestWeightedQuantile_UnequalWeights_InterpolatesEmpiricalCDF(t *testing.T) {
	t.Parallel()

	// Given samples whose sorted cumulative weights are 0.25, 0.75, and 1.
	samples := []float64{3, 1, 2}
	weights := []float64{1, 1, 2}
	queries := []float64{0, 0.1, 0.25, 0.5, 0.875, 1}

	// When
	got := quantile.WeightedQuantile(queries, weights, samples)

	// Then
	assertClose(t, "quantiles", got, []float64{1, 1, 1, 1.5, 2.5, 3})
}

// TestWeightedQuantile_NaNSample_SortsLast verifies that WeightedQuantile, given NaN sample, sorts last.
func TestWeightedQuantile_NaNSample_SortsLast(t *testing.T) {
	t.Parallel()

	// Given a NaN, which torch.sort places after every number.
	samples := []float32{float32(math.NaN()), 1}
	weights := []float64{1, 1}

	// When
	got := quantile.WeightedQuantile([]float64{0.25}, weights, samples)

	// Then
	assertClose(t, "quantiles", got, []float32{1})
}

// TestWeightedQuantile_InvalidArguments_Panics verifies that WeightedQuantile, given invalid arguments, panics.
func TestWeightedQuantile_InvalidArguments_Panics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		queries []float64
		weights []float64
		samples []float64
	}{
		{"no samples", []float64{0.5}, nil, nil},
		{"weights and samples differ", []float64{0.5}, []float64{1}, []float64{1, 2}},
		{"zero weight", []float64{0.5}, []float64{1, 0}, []float64{1, 2}},
		{"NaN weight", []float64{0.5}, []float64{1, nan()}, []float64{1, 2}},
		{"query below zero", []float64{-0.5}, []float64{1}, []float64{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt

			// When
			recovered := panicValue(func() { quantile.WeightedQuantile(tt.queries, tt.weights, tt.samples) })

			// Then
			if recovered == nil {
				t.Errorf("WeightedQuantile() did not panic")
			}
		})
	}
}

// nan returns a float64 NaN for table entries.
func nan() float64 { return math.NaN() }

// formatLevel renders a level for a test label.
func formatLevel(level float64) string { return strconv.FormatFloat(level, 'g', -1, 64) }

// column returns the first step of each line in lines.
func column[T float32 | float64](lines [][]T) []T {
	first := make([]T, len(lines))
	for i, line := range lines {
		first[i] = line[0]
	}
	return first
}
