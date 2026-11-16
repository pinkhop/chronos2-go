package batching_test

import (
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// TestPlan_TaskWithCovariates_OrdersTargetsThenPastOnlyThenKnownFutureByName verifies that a Plan orders targets, then past-only, then known-future covariates by name.
func TestPlan_TaskWithCovariates_OrdersTargetsThenPastOnlyThenKnownFutureByName(t *testing.T) {
	t.Parallel()

	// Given a two-target task whose numeric and categorical covariates
	// interleave alphabetically across the past-only and known-future classes
	task := batching.Task{
		Targets: [][]float64{{1, 1}, {2, 2}},
		PastNumeric: map[string][]float64{
			"zeta": {30, 30},
			"beta": {40, 40},
		},
		FutureNumeric: map[string][]float64{"beta": make([]float64, 16)},
		PastCategorical: map[string][]string{
			"alpha": {"x", "x"},
			"gamma": {"y", "y"},
		},
		FutureCategorical: map[string][]string{"gamma": slices.Repeat([]string{"y"}, 16)},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then rows are targets, past-only (alpha, zeta), then known-future (beta, gamma)
	if len(batches) != 1 || batches[0].Rows != 6 {
		t.Fatalf("batches = %d, rows = %v, want one batch of 6 rows", len(batches), rowCounts(batches))
	}
	batch := batches[0]
	want := [][]float32{{1, 1}, {2, 2}, {0, 0}, {30, 30}, {40, 40}, {0, 0}}
	for r, row := range want {
		assertRow(t, "context row", contextRow(batch, r), row)
	}
}

// TestPlan_NumericCovariates_ConvertedToFloat32WithMissingValuesKept verifies that a Plan converts numeric covariates to float32, keeping missing values.
func TestPlan_NumericCovariates_ConvertedToFloat32WithMissingValuesKept(t *testing.T) {
	t.Parallel()

	// Given a task with a past-only numeric covariate holding a missing value
	task := batching.Task{
		Targets:     [][]float64{{1, 2, 3}},
		PastNumeric: map[string][]float64{"price": {0.1, nan64, 2.5}},
	}

	// When the task is built
	batches := buildAll([]batching.Task{task}, defaultOptions())

	// Then the covariate row holds the float32 values with NaN preserved
	assertRow(t, "covariate row", contextRow(batches[0], 1), []float32{0.1, nan, 2.5})
}

// TestPlan_FutureRows_TargetsAndPastOnlyAreUnknownAndKnownFutureCarriesValues verifies that a Plan leaves target and past-only future rows unknown while known-future rows carry their values.
func TestPlan_FutureRows_TargetsAndPastOnlyAreUnknownAndKnownFutureCarriesValues(t *testing.T) {
	t.Parallel()

	// Given a 20-step horizon and a task with a past-only covariate and a
	// known-future covariate whose second future value is missing
	future := make([]float64, 20)
	for i := range future {
		future[i] = float64(i + 1)
	}
	future[1] = nan64
	task := batching.Task{
		Targets:       [][]float64{{1, 2}},
		PastNumeric:   map[string][]float64{"past": {3, 4}, "known": {5, 6}},
		FutureNumeric: map[string][]float64{"known": future},
	}
	options := defaultOptions()
	options.Horizon = 20

	// When the task is built
	batches := buildAll([]batching.Task{task}, options)

	// Then the future length rounds up to two patches, the target and past-only
	// rows are all NaN, and the known-future row carries its values then NaN
	batch := batches[0]
	if batch.FutureLength != 32 {
		t.Fatalf("FutureLength = %d, want 32", batch.FutureLength)
	}
	wantKnown := make([]float32, 0, 32)
	for _, value := range future {
		wantKnown = append(wantKnown, float32(value))
	}
	wantKnown = append(wantKnown, nanFloats(12)...)
	assertRow(t, "target future", futureRow(batch, 0), nanFloats(32))
	assertRow(t, "past-only future", futureRow(batch, 1), nanFloats(32))
	assertRow(t, "known-future future", futureRow(batch, 2), wantKnown)
}

// TestPlan_LongHorizon_KeepsFullFutureLength verifies that a Plan, given long horizon, keeps full future length.
func TestPlan_LongHorizon_KeepsFullFutureLength(t *testing.T) {
	t.Parallel()

	// Given a horizon beyond one engine pass
	options := defaultOptions()
	options.Horizon = 1500

	// When a univariate task is built
	batches := buildAll([]batching.Task{univariate(1, 2)}, options)

	// Then the future spans the whole horizon rounded up to a patch
	if batches[0].FutureLength != 1504 || len(batches[0].Future) != 1504 {
		t.Errorf("FutureLength = %d, len(Future) = %d, want 1504", batches[0].FutureLength, len(batches[0].Future))
	}
}

// TestPlan_LongHistory_TrimmedToMostRecentContextLengthSteps verifies that a Plan trims a long history to its most recent ContextLength steps.
func TestPlan_LongHistory_TrimmedToMostRecentContextLengthSteps(t *testing.T) {
	t.Parallel()

	// Given a three-step window and histories of five and two steps
	options := defaultOptions()
	options.ContextLength = 3
	tasks := []batching.Task{univariate(1, 2, 3, 4, 5), univariate(8, 9)}

	// When the tasks are built into one batch
	batches := buildAll(tasks, options)

	// Then the long row keeps its last three steps and the short row is
	// left-padded with NaN to the batch's longest row
	batch := batches[0]
	if batch.ContextLength != 3 {
		t.Fatalf("ContextLength = %d, want 3", batch.ContextLength)
	}
	assertRow(t, "trimmed row", contextRow(batch, 0), []float32{3, 4, 5})
	assertRow(t, "padded row", contextRow(batch, 1), []float32{nan, 8, 9})
}

// TestPlan_ShortHistories_BatchContextLengthIsLongestRow verifies that a Plan, given short histories, batch context length is longest row.
func TestPlan_ShortHistories_BatchContextLengthIsLongestRow(t *testing.T) {
	t.Parallel()

	// Given histories of four and two steps, both within the window
	tasks := []batching.Task{univariate(1, 2, 3, 4), univariate(5, 6)}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then the batch is only as long as its longest row
	if batches[0].ContextLength != 4 || len(batches[0].Context) != 8 {
		t.Errorf("ContextLength = %d, len(Context) = %d, want 4 and 8", batches[0].ContextLength, len(batches[0].Context))
	}
}

// TestPlan_SeveralTasks_EachTaskGetsItsOwnGroup verifies that a Plan, given several tasks, each task gets its own group.
func TestPlan_SeveralTasks_EachTaskGetsItsOwnGroup(t *testing.T) {
	t.Parallel()

	// Given a two-target task followed by a univariate task
	tasks := []batching.Task{
		{Targets: [][]float64{{1}, {2}}},
		univariate(3),
	}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then the first task's rows share one group and the second task has another
	if got := batches[0].GroupIDs; !slices.Equal(got, []int{0, 0, 1}) {
		t.Errorf("GroupIDs = %v, want [0 0 1]", got)
	}
}

// TestPlan_CrossLearning_AllRowsShareGroupZero verifies that a Plan, given cross learning, all rows share group zero.
func TestPlan_CrossLearning_AllRowsShareGroupZero(t *testing.T) {
	t.Parallel()

	// Given cross-learning over a two-target task and a univariate task
	tasks := []batching.Task{
		{Targets: [][]float64{{1}, {2}}},
		univariate(3),
	}
	options := defaultOptions()
	options.CrossLearning = true

	// When the tasks are built
	batches := buildAll(tasks, options)

	// Then every row belongs to group 0
	if got := batches[0].GroupIDs; !slices.Equal(got, []int{0, 0, 0}) {
		t.Errorf("GroupIDs = %v, want [0 0 0]", got)
	}
}

// TestPlan_TasksWithCovariates_TargetRefsLocateEachTargetRow verifies that a Plan, given tasks with covariates, target refs locate each target row.
func TestPlan_TasksWithCovariates_TargetRefsLocateEachTargetRow(t *testing.T) {
	t.Parallel()

	// Given a univariate task with one covariate followed by a two-target task
	tasks := []batching.Task{
		{Targets: [][]float64{{1}}, PastNumeric: map[string][]float64{"c": {9}}},
		{Targets: [][]float64{{2}, {3}}},
	}

	// When the tasks are built
	batches := buildAll(tasks, defaultOptions())

	// Then the refs skip the covariate row and number targets within each task
	want := []batching.TargetRef{
		{Task: 0, Target: 0, Row: 0},
		{Task: 1, Target: 0, Row: 2},
		{Task: 1, Target: 1, Row: 3},
	}
	if got := batches[0].Targets; !slices.Equal(got, want) {
		t.Errorf("Targets = %v, want %v", got, want)
	}
}

// TestPlan_NoTasks_ReturnsNoBatches verifies that a Plan, given no tasks, returns no batches.
func TestPlan_NoTasks_ReturnsNoBatches(t *testing.T) {
	t.Parallel()

	// Given no tasks
	var tasks []batching.Task

	// When they are built
	batches := buildAll(tasks, defaultOptions())

	// Then there are no batches
	if len(batches) != 0 {
		t.Errorf("len(batches) = %d, want 0", len(batches))
	}
}
