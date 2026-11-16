package testmodel

import (
	"fmt"
	"math"
)

// FiniteLines returns an error unless every value of lines is finite.
// Parity harnesses use it on a known deviation, a task exempt from comparison
// with the reference, to check that its forecast is still well formed.
func FiniteLines(lines [][]float64) error {
	for level, line := range lines {
		for step, value := range line {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("level %d step %d is %v, want a finite value", level, step, value)
			}
		}
	}
	return nil
}

// OrderedLines returns an error unless, at every step, no level's value is
// below the previous level's. lines holds one line per level, in increasing
// level order, each with one value per step, and every line must have the
// same length. Parity harnesses use it on a trained checkpoint's known
// deviations; a randomly initialized checkpoint does not order its levels.
func OrderedLines(lines [][]float64) error {
	for level := 1; level < len(lines); level++ {
		for step, value := range lines[level] {
			if value < lines[level-1][step] {
				return fmt.Errorf("level %d step %d is %v, below level %d's %v",
					level, step, value, level-1, lines[level-1][step])
			}
		}
	}
	return nil
}

// Deviation summarizes how one target's forecast lines differ from the
// reference.
type Deviation struct {
	// Outside counts the values outside the tolerance.
	Outside int

	// Total counts the values compared.
	Total int

	// MaxError is the largest |got − want| among the compared values; pairs
	// involving NaN do not count.
	MaxError float64

	// First describes the first value outside the tolerance; it is empty when
	// every value is within it.
	First string
}

// MeasureLines compares lines against target targetIndex of want, a
// [targets, levels, horizon] reference tensor, value by value under Within's
// combined tolerance with the given abs and rel. lines holds one line per
// level, each with one value per step, and must match want's levels and
// horizon; harnesses check the shape before measuring.
func MeasureLines(lines [][]float64, want Tensor, targetIndex int, abs, rel float64) Deviation {
	levels, horizon := want.Shape[1], want.Shape[2]
	deviation := Deviation{Total: levels * horizon}
	for level, line := range lines {
		for step, got := range line {
			expected := float64(want.Values[(targetIndex*levels+level)*horizon+step])
			if difference := math.Abs(got - expected); !math.IsNaN(difference) {
				deviation.MaxError = max(deviation.MaxError, difference)
			}
			if Within(got, expected, abs, rel) {
				continue
			}
			if deviation.Outside == 0 {
				deviation.First = fmt.Sprintf("level %d step %d = %v, want %v", level, step, got, expected)
			}
			deviation.Outside++
		}
	}
	return deviation
}
