// Package quantile reproduces the reference implementation's quantile
// arithmetic: linear interpolation across quantile levels, weighted
// quantiles of an empirical distribution, and the probability mass each
// quantile level represents.
//
// Every function computes in float32 whatever its input type, because the
// reference casts levels, values, and samples to float32 before computing;
// float64 results are float32 results widened. Arguments that violate a
// documented precondition cause a panic, since the root package validates
// all caller input before reaching this package.
package quantile

import (
	"fmt"
	"math"
	"sort"
)

// bracket locates one query level between two padded levels.
type bracket struct {
	// lower is the padded index of the bracket's lower level.
	lower int

	// upper is the padded index of the bracket's upper level.
	upper int

	// weight is the query's fractional position from lower to upper.
	weight float32
}

// padding records how the reference extends sorted levels to cover [0, 1].
type padding struct {
	// levels holds the sorted levels with 0 prepended when the smallest level
	// exceeds 0 and 1 appended when the largest falls short of 1.
	levels []float32

	// leading is 1 when a 0 was prepended and 0 otherwise.
	leading int

	// count is the number of original levels.
	count int
}

// padLevels returns sorted, non-empty levels in [0, 1] extended with 0 and 1
// as the reference does, so every query in [0, 1] falls within a bracket.
func padLevels(sorted []float32) padding {
	padded := padding{count: len(sorted)}
	if sorted[0] > 0 {
		padded.levels = append(padded.levels, 0)
		padded.leading = 1
	}
	padded.levels = append(padded.levels, sorted...)
	if sorted[len(sorted)-1] < 1 {
		padded.levels = append(padded.levels, 1)
	}
	return padded
}

// source maps a padded index to the index of the original level whose value
// it carries: a prepended 0 repeats the first value and an appended 1 the
// last.
func (p padding) source(paddedIndex int) int {
	return min(max(paddedIndex-p.leading, 0), p.count-1)
}

// locate returns the bracket of query within p, reproducing the reference's
// searchsorted with right=True, the clamp of the upper index to the last
// level, and nan_to_num on the interpolation weight. query must lie in
// [0, 1].
func (p padding) locate(query float32) bracket {
	levels := p.levels
	upper := sort.Search(len(levels), func(i int) bool { return levels[i] > query })
	upper = min(upper, len(levels)-1)
	lower := upper - 1
	weight := (query - levels[lower]) / (levels[upper] - levels[lower])
	return bracket{lower: lower, upper: upper, weight: nanToNumber(weight)}
}

// nanToNumber mirrors torch.nan_to_num with nan=0: NaN becomes 0 and an
// infinity becomes the largest finite float32 of its sign. A NaN weight
// arises when a query sits on two tied levels.
func nanToNumber(value float32) float32 {
	switch {
	case value != value:
		return 0
	case value > math.MaxFloat32:
		return math.MaxFloat32
	case value < -math.MaxFloat32:
		return -math.MaxFloat32
	default:
		return value
	}
}

// lerp interpolates from lower to upper by weight in float32, in the
// reference's operation order.
func lerp(lower, upper, weight float32) float32 {
	return lower + weight*(upper-lower)
}

// toLevels converts probability levels to float32, panicking with a message
// naming role when any is outside [0, 1] or NaN.
func toLevels(role string, levels []float64) []float32 {
	converted := make([]float32, len(levels))
	for i, level := range levels {
		if !(level >= 0 && level <= 1) {
			panic(fmt.Sprintf("quantile: %s level %v outside [0, 1]", role, level))
		}
		converted[i] = float32(level)
	}
	return converted
}
