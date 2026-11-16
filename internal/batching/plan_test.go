package batching_test

import (
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// TestNewPlan_TasksExceedingBatchSize_CountsBatches verifies that NewPlan
// reports how many batches the call needs, packing tasks in order without
// overshooting the batch size.
func TestNewPlan_TasksExceedingBatchSize_CountsBatches(t *testing.T) {
	t.Parallel()

	// Given a three-row limit and tasks of 2, 1, 2, and 3 rows
	options := defaultOptions()
	options.BatchSize = 3

	// When the tasks are planned
	plan := batching.NewPlan(tasksWithRows(2, 1, 2, 3), options)

	// Then they need three batches
	if got := plan.Len(); got != 3 {
		t.Errorf("Len() = %d, want 3", got)
	}
}

// TestNewPlan_NoTasks_HasNoBatches verifies that a plan over no tasks has no
// batches.
func TestNewPlan_NoTasks_HasNoBatches(t *testing.T) {
	t.Parallel()

	// Given default options
	options := defaultOptions()

	// When a plan is made over no tasks
	plan := batching.NewPlan(nil, options)

	// Then it has no batches
	if got := plan.Len(); got != 0 {
		t.Errorf("Len() = %d, want 0", got)
	}
}

// TestPlan_Batch_LaterBatchBuiltAlone_UsesCallWideOrdinalCodes verifies that
// building only a later batch still encodes categories with codes drawn from
// every task in the call, so a batch's contents never depend on which other
// batches were built.
func TestPlan_Batch_LaterBatchBuiltAlone_UsesCallWideOrdinalCodes(t *testing.T) {
	t.Parallel()

	// Given two multi-target tasks in separate batches whose categories
	// together sort as blue, green, red
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
	options := defaultOptions()
	options.BatchSize = 3
	plan := batching.NewPlan(tasks, options)
	if plan.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", plan.Len())
	}

	// When only the second batch is built
	second := plan.Batch(1)

	// Then the second task's covariate row uses codes that include the first
	// task's "blue"
	assertRow(t, "second task", contextRow(second, 2), []float32{1, 2, 2})
	if got := taskIndexes(second); len(got) != 1 || got[0] != 1 {
		t.Errorf("batch 1 tasks = %v, want [1]", got)
	}
}
