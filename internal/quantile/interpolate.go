package quantile

import (
	"cmp"
	"fmt"
	"slices"
)

// Interpolate returns the values at queryLevels of quantile lines known at
// levels, interpolating linearly between neighboring levels independently at
// each step. It reproduces the reference's interpolate_quantiles, which
// Forecast.Percentile and the long-horizon unroll both rely on.
//
// values[i] is the line at levels[i], and every line has the same number of
// steps. levels need not be sorted. A query below the smallest level takes
// the smallest level's line, and one above the largest level takes the
// largest level's line, because the reference pads the levels with 0 and 1.
//
// The result holds one line per query, indexed [query][step]. It panics when
// levels is empty, its length differs from values, the lines differ in
// length, or any level or query lies outside [0, 1]. Interpolate does not
// modify its arguments and is safe for concurrent use.
func Interpolate[T float32 | float64](queryLevels, levels []float64, values [][]T) [][]T {
	if len(levels) == 0 || len(levels) != len(values) {
		panic(fmt.Sprintf("quantile: Interpolate: %d levels for %d lines", len(levels), len(values)))
	}
	steps := len(values[0])
	for _, line := range values {
		if len(line) != steps {
			panic(fmt.Sprintf("quantile: Interpolate: lines of %d and %d steps", steps, len(line)))
		}
	}
	queries := toLevels("query", queryLevels)

	// Sort the levels, remembering which line each sorted level owns.
	converted := toLevels("original", levels)
	order := make([]int, len(converted))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(converted[a], converted[b]) })
	sorted := make([]float32, len(order))
	for i, original := range order {
		sorted[i] = converted[original]
	}
	padded := padLevels(sorted)

	// Bracket each query once; the levels are shared by every step.
	result := make([][]T, len(queries))
	for q, query := range queries {
		position := padded.locate(query)
		lowerLine := values[order[padded.source(position.lower)]]
		upperLine := values[order[padded.source(position.upper)]]
		line := make([]T, steps)
		for step := range steps {
			line[step] = T(lerp(float32(lowerLine[step]), float32(upperLine[step]), position.weight))
		}
		result[q] = line
	}
	return result
}
