package batching

import (
	"maps"
	"math"
	"slices"
)

// missingCategory is the categorical value that marks a missing observation.
// It is encoded as a category of its own, never masked, matching the
// reference's treatment of a missing (NaN) categorical value.
const missingCategory = ""

// targetEncodingSmoothing is the weight given to the target's overall mean
// when smoothing a category's mean, matching the reference's smooth=1.
const targetEncodingSmoothing = 1.0

// ordinalCategories maps each categorical covariate name to its ordinal codes:
// the named categories seen in any task's history, sorted lexicographically and
// numbered from 0, with the missing category taking the code after them.
type ordinalCategories map[string]map[string]int

// newOrdinalCategories collects the ordinal codes of every categorical
// covariate across all tasks' histories. Future values never add categories.
func newOrdinalCategories(tasks []Task) ordinalCategories {
	named := make(map[string]map[string]struct{})
	for _, task := range tasks {
		for name, values := range task.PastCategorical {
			set, ok := named[name]
			if !ok {
				set = make(map[string]struct{})
				named[name] = set
			}
			for _, value := range values {
				if value != missingCategory {
					set[value] = struct{}{}
				}
			}
		}
	}

	categories := make(ordinalCategories, len(named))
	for name, set := range named {
		sorted := slices.Sorted(maps.Keys(set))
		codes := make(map[string]int, len(sorted)+1)
		for code, category := range sorted {
			codes[category] = code
		}
		codes[missingCategory] = len(sorted)
		categories[name] = codes
	}
	return categories
}

// encodeCategorical encodes task's categorical covariate name, returning its
// past row and its future row (nil when the covariate is past-only). A task with
// one target uses target encoding; a task with several uses ordinal codes from
// categories, with a future category absent from every history encoded as NaN
// (unknown).
func encodeCategorical(task Task, name string, categories ordinalCategories) (past, future []float32) {
	pastValues := task.PastCategorical[name]
	futureValues := task.FutureCategorical[name]

	var encode func(category string) float32
	if len(task.Targets) == 1 {
		encode = newTargetEncoder(task.Targets[0], pastValues).encode
	} else {
		codes := categories[name]
		encode = func(category string) float32 {
			code, ok := codes[category]
			if !ok {
				return float32(math.NaN())
			}
			return float32(code)
		}
	}
	return mapCategories(pastValues, encode), mapCategories(futureValues, encode)
}

// mapCategories applies encode to each value, returning nil for an empty or
// nil slice.
func mapCategories(values []string, encode func(category string) float32) []float32 {
	if len(values) == 0 {
		return nil
	}
	encoded := make([]float32, len(values))
	for i, value := range values {
		encoded[i] = encode(value)
	}
	return encoded
}

// targetEncoder maps categories to smoothed means of one target's observed
// values, as the reference's per-item target encoding does.
type targetEncoder struct {
	// mean is the mean of the target's observed values, or 0 when none are
	// observed.
	mean float64

	// encoded maps each category observed alongside at least one observed
	// target value to (smoothing × mean + sum) / (smoothing + count).
	encoded map[string]float32
}

// newTargetEncoder computes target-encoding statistics from the full target
// history and the category at each of its steps; categories has the target's
// length. Steps whose target is missing contribute to no statistic.
//
// Each target value is rounded to float32 before being accumulated in float64,
// because the reference converts the target to float32 before summing it in
// float64; this reproduces its encoded values exactly.
func newTargetEncoder(target []float64, categories []string) targetEncoder {
	var total float64
	var count int
	sums := make(map[string]float64)
	counts := make(map[string]int)
	for i, value := range target {
		observed := float64(float32(value))
		if math.IsNaN(observed) || math.IsInf(observed, 0) {
			continue
		}
		total += observed
		count++
		sums[categories[i]] += observed
		counts[categories[i]]++
	}

	var mean float64
	if count > 0 {
		mean = total / float64(count)
	}
	encoded := make(map[string]float32, len(sums))
	for category, sum := range sums {
		smoothed := (targetEncodingSmoothing*mean + sum) / (targetEncodingSmoothing + float64(counts[category]))
		encoded[category] = float32(smoothed)
	}
	return targetEncoder{mean: mean, encoded: encoded}
}

// encode returns category's target encoding. A category never observed
// alongside an observed target value, including one seen only in the future,
// encodes to the target's mean, as the smoothing formula gives with a zero
// count.
func (e targetEncoder) encode(category string) float32 {
	if value, ok := e.encoded[category]; ok {
		return value
	}
	return float32(e.mean)
}
