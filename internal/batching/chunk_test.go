package batching_test

import (
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// tasksWithRows returns one task per entry of rows, each with that many
// one-step targets, so a task contributes exactly that many engine rows.
func tasksWithRows(rows ...int) []batching.Task {
	tasks := make([]batching.Task, len(rows))
	for i, n := range rows {
		targets := make([][]float64, n)
		for j := range targets {
			targets[j] = []float64{float64(i)}
		}
		tasks[i] = batching.Task{Targets: targets}
	}
	return tasks
}

// taskIndexes returns the distinct task index of each batch's target refs, in
// order of first appearance.
func taskIndexes(batch batching.Batch) []int {
	var indexes []int
	for _, ref := range batch.Targets {
		if !slices.Contains(indexes, ref.Task) {
			indexes = append(indexes, ref.Task)
		}
	}
	return indexes
}

// TestPlan_TasksExceedingBatchSize_PackedInOrderWithoutOvershooting verifies that a Plan packs tasks in order without exceeding the batch size.
func TestPlan_TasksExceedingBatchSize_PackedInOrderWithoutOvershooting(t *testing.T) {
	t.Parallel()

	// Given a three-row limit and tasks of 2, 1, 2, and 3 rows
	options := defaultOptions()
	options.BatchSize = 3

	// When the tasks are built
	batches := buildAll(tasksWithRows(2, 1, 2, 3), options)

	// Then tasks fill batches in order and no batch exceeds three rows
	if got := rowCounts(batches); !slices.Equal(got, []int{3, 2, 3}) {
		t.Fatalf("row counts = %v, want [3 2 3]", got)
	}
	want := [][]int{{0, 1}, {2}, {3}}
	for i, batch := range batches {
		if got := taskIndexes(batch); !slices.Equal(got, want[i]) {
			t.Errorf("batch %d tasks = %v, want %v", i, got, want[i])
		}
	}
}

// TestPlan_TasksFillingBatchExactly_StartNewBatchAfterward verifies that a Plan starts a new batch after tasks fill one exactly.
func TestPlan_TasksFillingBatchExactly_StartNewBatchAfterward(t *testing.T) {
	t.Parallel()

	// Given a four-row limit and tasks of 2, 2, and 1 rows
	options := defaultOptions()
	options.BatchSize = 4

	// When the tasks are built
	batches := buildAll(tasksWithRows(2, 2, 1), options)

	// Then the first two tasks fill one batch and the last task starts another
	if got := rowCounts(batches); !slices.Equal(got, []int{4, 1}) {
		t.Errorf("row counts = %v, want [4 1]", got)
	}
}

// TestPlan_OversizedTask_FormsBatchOfItsOwnWithoutSplitting verifies that a Plan, given oversized task, forms batch of its own without splitting.
func TestPlan_OversizedTask_FormsBatchOfItsOwnWithoutSplitting(t *testing.T) {
	t.Parallel()

	// Given a two-row limit and tasks of 1, 3, and 1 rows
	options := defaultOptions()
	options.BatchSize = 2

	// When the tasks are built
	batches := buildAll(tasksWithRows(1, 3, 1), options)

	// Then the three-row task occupies a batch alone and its neighbors do not
	// join it
	if got := rowCounts(batches); !slices.Equal(got, []int{1, 3, 1}) {
		t.Errorf("row counts = %v, want [1 3 1]", got)
	}
}

// TestPlan_LaterBatch_GroupIDsAndTargetRefsAreBatchLocal verifies that a Plan numbers a later batch's groups and rows from zero.
func TestPlan_LaterBatch_GroupIDsAndTargetRefsAreBatchLocal(t *testing.T) {
	t.Parallel()

	// Given a two-row limit and three univariate tasks
	options := defaultOptions()
	options.BatchSize = 2

	// When the tasks are built
	batches := buildAll(tasksWithRows(1, 1, 1), options)

	// Then the second batch numbers its groups and rows from zero while its
	// refs keep the call-wide task index
	if len(batches) != 2 {
		t.Fatalf("len(batches) = %d, want 2", len(batches))
	}
	if got := batches[1].GroupIDs; !slices.Equal(got, []int{0}) {
		t.Errorf("GroupIDs = %v, want [0]", got)
	}
	want := []batching.TargetRef{{Task: 2, Target: 0, Row: 0}}
	if got := batches[1].Targets; !slices.Equal(got, want) {
		t.Errorf("Targets = %v, want %v", got, want)
	}
}
