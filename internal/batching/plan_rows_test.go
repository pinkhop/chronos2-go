package batching_test

import (
	"slices"
	"sync"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// covariateTask returns a one-target task with one row of each covariate
// class: a past-only numeric, a known-future numeric, a past-only
// categorical, and a known-future categorical, five rows in all.
func covariateTask() batching.Task {
	return batching.Task{
		Targets:           [][]float64{{1, 2}},
		PastNumeric:       map[string][]float64{"load": {3, 4}, "price": {5, 6}},
		FutureNumeric:     map[string][]float64{"price": {7}},
		PastCategorical:   map[string][]string{"day": {"mon", "tue"}, "promo": {"", "yes"}},
		FutureCategorical: map[string][]string{"promo": {"yes"}},
	}
}

// TestNewPlan_TasksWithEveryCovariateClass_ChunksByFullRowCount verifies that
// chunking counts a row for every covariate class, numeric and categorical,
// past-only and known-future, so batches pack exactly to the row limit.
func TestNewPlan_TasksWithEveryCovariateClass_ChunksByFullRowCount(t *testing.T) {
	t.Parallel()

	// Given three five-row tasks and a ten-row limit
	options := defaultOptions()
	options.BatchSize = 10
	tasks := []batching.Task{covariateTask(), covariateTask(), covariateTask()}

	// When every batch is built
	batches := buildAll(tasks, options)

	// Then two tasks fill the first batch and the third forms the second
	if got := rowCounts(batches); !slices.Equal(got, []int{10, 5}) {
		t.Errorf("row counts = %v, want [10 5]", got)
	}
}

// TestPlan_Batch_ConcurrentBuilds_MatchSequentialBuilds verifies that a Plan
// shared by goroutines builds the same batches as sequential calls, so
// concurrent use (checked under -race) reads only immutable state.
func TestPlan_Batch_ConcurrentBuilds_MatchSequentialBuilds(t *testing.T) {
	t.Parallel()

	// Given a plan of several multi-target batches and their sequential builds
	multiTarget := covariateTask()
	multiTarget.Targets = [][]float64{{1, 2}, {3, 4}}
	options := defaultOptions()
	options.BatchSize = 6
	tasks := []batching.Task{multiTarget, covariateTask(), multiTarget, covariateTask()}
	plan := batching.NewPlan(tasks, options)
	sequential := buildAll(tasks, options)
	if len(sequential) < 2 {
		t.Fatalf("got %d batches, want at least 2", len(sequential))
	}

	// When every batch is built twice at once from the shared plan
	concurrent := make([]batching.Batch, 2*plan.Len())
	var group sync.WaitGroup
	for i := range concurrent {
		group.Go(func() { concurrent[i] = plan.Batch(i % plan.Len()) })
	}
	group.Wait()

	// Then each concurrent build equals its sequential counterpart
	for i, batch := range concurrent {
		if !equalBatches(sequential[i%len(sequential)], batch) {
			t.Errorf("concurrent build %d of batch %d differs from its sequential build", i, i%len(sequential))
		}
	}
}

// equalBatches reports whether a and b hold the same shape, values, groups,
// and target refs, treating any two NaN values as equal.
func equalBatches(a, b batching.Batch) bool {
	return a.Rows == b.Rows && a.ContextLength == b.ContextLength && a.FutureLength == b.FutureLength &&
		equalFloats(a.Context, b.Context) && equalFloats(a.Future, b.Future) &&
		slices.Equal(a.GroupIDs, b.GroupIDs) && slices.Equal(a.Targets, b.Targets)
}
