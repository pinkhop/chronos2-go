package testmodel

import (
	"math"
	"testing"
)

// TestFiniteLines_Lines_RejectsOnlyNonFiniteValues verifies that FiniteLines
// accepts finite lines, even when levels cross, and rejects lines holding an
// infinity or a NaN.
func TestFiniteLines_Lines_RejectsOnlyNonFiniteValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		// name describes the case.
		name string

		// lines holds one line per level, each one value per step.
		lines [][]float64

		// wantErr reports whether FiniteLines should reject lines.
		wantErr bool
	}{
		{name: "finite with crossing levels", lines: [][]float64{{1, 2}, {0.5, 3}}},
		{name: "infinite value", lines: [][]float64{{1, 2}, {math.Inf(-1), 3}}, wantErr: true},
		{name: "NaN value", lines: [][]float64{{1, math.NaN()}, {2, 3}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt.lines

			// When
			err := FiniteLines(tt.lines)

			// Then
			if (err != nil) != tt.wantErr {
				t.Errorf("FiniteLines() error = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

// TestOrderedLines_Lines_RejectsOnlyCrossingLevels verifies that OrderedLines
// accepts lines that never decrease from one level to the next, including
// ties, and rejects lines where a level falls below the previous one.
func TestOrderedLines_Lines_RejectsOnlyCrossingLevels(t *testing.T) {
	t.Parallel()

	tests := []struct {
		// name describes the case.
		name string

		// lines holds one line per level, each one value per step.
		lines [][]float64

		// wantErr reports whether OrderedLines should reject lines.
		wantErr bool
	}{
		{name: "non-decreasing with ties", lines: [][]float64{{1, 2}, {1, 3}, {4, 3}}},
		{name: "crossing levels", lines: [][]float64{{1, 2}, {0.5, 3}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt.lines

			// When
			err := OrderedLines(tt.lines)

			// Then
			if (err != nil) != tt.wantErr {
				t.Errorf("OrderedLines() error = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

// TestMeasureLines_OneValueOutsideTolerance_CountsItAndItsError verifies that
// MeasureLines counts the values outside the combined tolerance and reports
// the largest absolute error among all values.
func TestMeasureLines_OneValueOutsideTolerance_CountsItAndItsError(t *testing.T) {
	t.Parallel()

	// Given target 1 of a [2 targets, 2 levels, 2 steps] reference, and lines
	// with one value 0.5 off and one 1e-6 off
	want := Tensor{Shape: []int{2, 2, 2}, Values: []float32{0, 0, 0, 0, 1, 2, 3, 4}}
	lines := [][]float64{{1, 2.5}, {3 + 1e-6, 4}}

	// When
	deviation := MeasureLines(lines, want, 1, 1e-3, 0)

	// Then
	if deviation.Outside != 1 || deviation.Total != 4 || math.Abs(deviation.MaxError-0.5) > 1e-9 || deviation.First == "" {
		t.Errorf("MeasureLines() = %+v, want 1 of 4 outside, MaxError 0.5, and a First description", deviation)
	}
}
