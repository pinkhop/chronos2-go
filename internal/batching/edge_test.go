package batching_test

import (
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// TestPlan_TasksWithDifferentCovariateSets_EachTaskGetsOnlyItsOwnRows verifies
// that a Plan lays out tasks whose covariate sets differ, including nil and
// empty maps, with only each task's own rows and groups.
func TestPlan_TasksWithDifferentCovariateSets_EachTaskGetsOnlyItsOwnRows(t *testing.T) {
	t.Parallel()

	// Given a task with a numeric covariate, a task with empty (not nil) maps,
	// and a task with a categorical covariate
	tasks := []batching.Task{
		{Targets: [][]float64{{1, 2}}, PastNumeric: map[string][]float64{"x": {3, 4}}},
		{
			Targets:         [][]float64{{5}},
			PastNumeric:     map[string][]float64{},
			FutureNumeric:   map[string][]float64{},
			PastCategorical: map[string][]string{},
		},
		{Targets: [][]float64{{6, 7}}, PastCategorical: map[string][]string{"c": {"a", "b"}}},
	}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then the batch holds five rows, each task's rows share its own group, and
	// the targets sit at rows 0, 2, and 3
	batch := batches[0]
	wantTargets := []batching.TargetRef{{Task: 0, Target: 0, Row: 0}, {Task: 1, Target: 0, Row: 2}, {Task: 2, Target: 0, Row: 3}}
	if batch.Rows != 5 || !slices.Equal(batch.GroupIDs, []int{0, 0, 1, 2, 2}) || !slices.Equal(batch.Targets, wantTargets) {
		t.Errorf("Rows = %d, GroupIDs = %v, Targets = %v; want 5, [0 0 1 2 2], %v",
			batch.Rows, batch.GroupIDs, batch.Targets, wantTargets)
	}
}

// TestPlan_KnownFutureCovariateWithAllMissingPast_KeepsFutureValues verifies
// that a Plan keeps a known-future covariate's future values even when every
// value of its past is missing.
func TestPlan_KnownFutureCovariateWithAllMissingPast_KeepsFutureValues(t *testing.T) {
	t.Parallel()

	// Given a known-future covariate whose past is entirely NaN
	future := make([]float64, 16)
	for i := range future {
		future[i] = float64(i)
	}
	task := batching.Task{
		Targets:       [][]float64{{1, 2}},
		PastNumeric:   map[string][]float64{"known": {nan64, nan64}},
		FutureNumeric: map[string][]float64{"known": future},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then the covariate's context row is all NaN and its future row carries
	// the supplied values
	batch := batches[0]
	wantFuture := make([]float32, 16)
	for i, value := range future {
		wantFuture[i] = float32(value)
	}
	assertRow(t, "covariate context", contextRow(batch, 1), nanFloats(2))
	assertRow(t, "covariate future", futureRow(batch, 1), wantFuture)
}

// TestPlan_CrossLearningFillingBatchExactly_FormsOneBatchInGroupZero verifies
// that a Plan keeps a cross-learning call whose rows exactly fill the batch size
// in one batch with every row in group 0.
func TestPlan_CrossLearningFillingBatchExactly_FormsOneBatchInGroupZero(t *testing.T) {
	t.Parallel()

	// Given cross-learning over three univariate tasks and a batch size of 3
	tasks := []batching.Task{univariate(1), univariate(2), univariate(3)}
	options := defaultOptions()
	options.BatchSize = 3
	options.CrossLearning = true

	// When the tasks are built
	batches := buildAll(tasks, options)

	// Then one batch holds all three rows in group 0
	if len(batches) != 1 || !slices.Equal(batches[0].GroupIDs, []int{0, 0, 0}) {
		t.Errorf("batches = %v, want one batch with GroupIDs [0 0 0]", rowCounts(batches))
	}
}
