package batching_test

import (
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// The single-target fixtures below share one history. Its observed target
// values are 1, 3, 5, and 10 (mean 4.75), so with smoothing weight 1:
//
//	"a" at targets 1 and 10: (4.75 + 11) / 3 = 5.25
//	""  at target 3:         (4.75 + 3) / 2  = 3.875
//	"b" at target 5:         (4.75 + 5) / 2  = 4.875
//
// The step whose target is missing still receives its category's encoding but
// contributes nothing to any statistic.

// encodingTarget is the shared single-target history.
var encodingTarget = []float64{1, nan64, 3, 5, 10}

// TestPlan_SingleTarget_TargetEncodesPastCategoriesIncludingMissing verifies that a Plan target-encodes a single-target task's past categories, including the missing one.
func TestPlan_SingleTarget_TargetEncodesPastCategoriesIncludingMissing(t *testing.T) {
	t.Parallel()

	// Given a single-target task with a past-only categorical covariate that
	// uses the empty string as a category
	task := batching.Task{
		Targets:         [][]float64{encodingTarget},
		PastCategorical: map[string][]string{"store": {"a", "a", "", "b", "a"}},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then each step holds its category's smoothed target mean
	assertRow(t, "encoded row", contextRow(batches[0], 1), []float32{5.25, 5.25, 3.875, 4.875, 5.25})
}

// TestPlan_SingleTarget_CategorySeenOnlyAtMissingTargetsEncodesToMean verifies that a Plan, given single target, category seen only at missing targets encodes to mean.
func TestPlan_SingleTarget_CategorySeenOnlyAtMissingTargetsEncodesToMean(t *testing.T) {
	t.Parallel()

	// Given a category "d" that occurs only where the target is missing
	task := batching.Task{
		Targets:         [][]float64{encodingTarget},
		PastCategorical: map[string][]string{"store": {"a", "d", "", "b", "a"}},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then "d" encodes to the target's overall mean
	assertRow(t, "encoded row", contextRow(batches[0], 1), []float32{5.25, 4.75, 3.875, 4.875, 5.25})
}

// TestPlan_SingleTarget_FutureCategoriesUsePastStatisticsAndUnseenGetMean verifies that a Plan, given single target, future categories use past statistics and unseen get mean.
func TestPlan_SingleTarget_FutureCategoriesUsePastStatisticsAndUnseenGetMean(t *testing.T) {
	t.Parallel()

	// Given known-future categories that are seen, unseen, and missing
	future := append([]string{"b", "c", "", "a"}, slices.Repeat([]string{"a"}, 12)...)
	task := batching.Task{
		Targets:           [][]float64{encodingTarget},
		PastCategorical:   map[string][]string{"store": {"a", "a", "", "b", "a"}},
		FutureCategorical: map[string][]string{"store": future},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then seen categories reuse their past encoding and the unseen "c" gets
	// the target's mean
	want := concat([]float32{4.875, 4.75, 3.875}, slices.Repeat([]float32{5.25}, 13))
	assertRow(t, "encoded future", futureRow(batches[0], 1), want)
}

// TestPlan_SingleTarget_EncodingStatisticsUseFullHistoryBeforeTrimming verifies that a Plan, given single target, encoding statistics use full history before trimming.
func TestPlan_SingleTarget_EncodingStatisticsUseFullHistoryBeforeTrimming(t *testing.T) {
	t.Parallel()

	// Given a two-step window over a four-step history whose target mean is 4
	task := batching.Task{
		Targets:         [][]float64{{1, 2, 3, 10}},
		PastCategorical: map[string][]string{"store": {"a", "b", "a", "b"}},
	}
	options := defaultOptions()
	options.ContextLength = 2

	// When the task is built
	batches := buildAll([]batching.Task{task}, options)

	// Then the kept steps use statistics from all four steps:
	// "a" = (4 + 4) / 3 and "b" = (4 + 12) / 3
	want := []float32{float32(8.0 / 3.0), float32(16.0 / 3.0)}
	assertRow(t, "encoded row", contextRow(batches[0], 1), want)
}

// TestPlan_MultipleTargets_OrdinalEncodesWithCallWideSortedCategories verifies that a Plan ordinal-encodes multi-target tasks with call-wide sorted categories.
func TestPlan_MultipleTargets_OrdinalEncodesWithCallWideSortedCategories(t *testing.T) {
	t.Parallel()

	// Given two multi-target tasks whose categories together sort as blue,
	// green, red, with the empty string taking the code after them
	tasks := []batching.Task{
		{
			Targets:         [][]float64{{1, 2, 3}, {4, 5, 6}},
			PastCategorical: map[string][]string{"color": {"red", "", "blue"}},
		},
		{
			Targets:         [][]float64{{1, 2, 3}, {4, 5, 6}},
			PastCategorical: map[string][]string{"color": {"green", "red", "red"}},
		},
	}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then each task's covariate row holds ordinal codes from the shared set
	assertRow(t, "first task", contextRow(batches[0], 2), []float32{2, 3, 0})
	assertRow(t, "second task", contextRow(batches[0], 5), []float32{1, 2, 2})
}

// TestPlan_MultipleTargets_FutureOrdinalCodesWithUnseenAsUnknown verifies that a Plan gives multi-target future categories ordinal codes, with unseen ones unknown.
func TestPlan_MultipleTargets_FutureOrdinalCodesWithUnseenAsUnknown(t *testing.T) {
	t.Parallel()

	// Given a known-future category seen only in another task's past, one seen
	// nowhere, and the empty string
	future := append([]string{"green", "purple", ""}, slices.Repeat([]string{"red"}, 13)...)
	tasks := []batching.Task{
		{
			Targets:           [][]float64{{1, 2, 3}, {4, 5, 6}},
			PastCategorical:   map[string][]string{"color": {"red", "", "blue"}},
			FutureCategorical: map[string][]string{"color": future},
		},
		{
			Targets:         [][]float64{{1, 2, 3}, {4, 5, 6}},
			PastCategorical: map[string][]string{"color": {"green", "red", "red"}},
		},
	}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then "green" takes its call-wide code, "purple" is unknown, and the
	// empty string takes the missing category's code
	want := concat([]float32{1, nan, 3}, slices.Repeat([]float32{2}, 13))
	assertRow(t, "encoded future", futureRow(batches[0], 2), want)
}

// TestPlan_MultipleTargets_OnlyMissingCategoryEncodesToZero verifies that a Plan encodes a covariate holding only the missing category to code zero.
func TestPlan_MultipleTargets_OnlyMissingCategoryEncodesToZero(t *testing.T) {
	t.Parallel()

	// Given a covariate whose every value is the missing category
	task := batching.Task{
		Targets:         [][]float64{{1, 2}, {3, 4}},
		PastCategorical: map[string][]string{"color": {"", ""}},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then the missing category is the only category and takes code 0
	assertRow(t, "encoded row", contextRow(batches[0], 2), []float32{0, 0})
}
