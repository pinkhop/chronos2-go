package quantile

import (
	"cmp"
	"fmt"
	"slices"
)

// WeightedQuantile returns the values at queryLevels of the empirical
// distribution that places weights[i] of probability mass on samples[i]. The
// long-horizon unroll uses it to reduce each step's predictions over many
// sample paths back to the model's native quantile levels.
//
// It reproduces the reference's weighted_quantile: the weights are
// normalized, the samples sorted with NaN last as torch.sort orders them, and
// the running sums of the sorted weights, clamped to [0, 1], become the
// levels passed to the same interpolation Interpolate performs. A query
// below the first running sum takes the smallest sample, and one above the
// last sum, when rounding leaves it short of 1, takes the largest.
//
// The result holds one value per query. It panics when samples is empty, its
// length differs from weights, any weight is not positive, or any query lies
// outside [0, 1]. WeightedQuantile does not modify its arguments and is safe
// for concurrent use.
func WeightedQuantile[T float32 | float64](queryLevels, weights []float64, samples []T) []T {
	if len(samples) == 0 || len(samples) != len(weights) {
		panic(fmt.Sprintf("quantile: WeightedQuantile: %d weights for %d samples", len(weights), len(samples)))
	}
	queries := toLevels("query", queryLevels)
	normalized := normalizeWeights(weights)

	// Sort the samples and carry each one's weight along.
	order := make([]int, len(samples))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return compareNaNLast(samples[a], samples[b]) })

	// Accumulate the sorted weights into the empirical CDF. torch.cumsum on
	// CPU accumulates float32 in float64 and rounds each running sum.
	cumulative := make([]float32, len(order))
	var running float64
	for i, original := range order {
		running += float64(normalized[original])
		cumulative[i] = min(max(float32(running), 0), 1)
	}
	padded := padLevels(cumulative)

	result := make([]T, len(queries))
	for q, query := range queries {
		position := padded.locate(query)
		lower := float32(samples[order[padded.source(position.lower)]])
		upper := float32(samples[order[padded.source(position.upper)]])
		result[q] = T(lerp(lower, upper, position.weight))
	}
	return result
}

// normalizeWeights converts weights to float32 and divides each by their
// total, panicking unless every weight is positive.
func normalizeWeights(weights []float64) []float32 {
	converted := make([]float32, len(weights))
	var total float64
	for i, weight := range weights {
		converted[i] = float32(weight)
		if !(converted[i] > 0) {
			panic(fmt.Sprintf("quantile: WeightedQuantile: weight %v is not positive", weight))
		}
		total += float64(converted[i])
	}
	total32 := float32(total)
	for i := range converted {
		converted[i] /= total32
	}
	return converted
}

// compareNaNLast orders numbers ascending with every NaN after them, as
// torch.sort does; cmp.Compare would put NaN first instead.
func compareNaNLast[T float32 | float64](a, b T) int {
	aNaN, bNaN := a != a, b != b
	switch {
	case aNaN && bNaN:
		return 0
	case aNaN:
		return 1
	case bNaN:
		return -1
	default:
		return cmp.Compare(a, b)
	}
}
