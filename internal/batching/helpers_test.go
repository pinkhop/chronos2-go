package batching_test

import (
	"math"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// nan is the float32 not-a-number value that marks missing, padded, and
// unknown values in engine batches.
var nan = float32(math.NaN())

// nan64 is the float64 not-a-number value that marks a missing observation in
// task inputs.
var nan64 = math.NaN()

// defaultOptions returns options that neither trim nor chunk small test inputs:
// a one-patch horizon, the full 8192-step window, and the default 256-row
// batch size.
func defaultOptions() batching.Options {
	return batching.Options{Horizon: 16, ContextLength: 8192, BatchSize: 256}
}

// univariate returns a task with a single target holding values and no
// covariates.
func univariate(values ...float64) batching.Task {
	return batching.Task{Targets: [][]float64{values}}
}

// contextRow returns row r of batch's context matrix, ContextLength values
// long. It shares memory with batch.Context.
func contextRow(batch batching.Batch, r int) []float32 {
	return batch.Context[r*batch.ContextLength : (r+1)*batch.ContextLength]
}

// futureRow returns row r of batch's future matrix, FutureLength values long.
// It shares memory with batch.Future.
func futureRow(batch batching.Batch, r int) []float32 {
	return batch.Future[r*batch.FutureLength : (r+1)*batch.FutureLength]
}

// equalFloats reports whether a and b hold the same values in the same order,
// treating any two NaN values as equal.
func equalFloats(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool {
		return x == y || (x != x && y != y)
	})
}

// nanFloats returns n NaN values.
func nanFloats(n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = nan
	}
	return values
}

// concat returns the concatenation of parts as a new slice.
func concat(parts ...[]float32) []float32 {
	return slices.Concat(parts...)
}

// rowCounts returns the Rows field of each batch, in order.
func rowCounts(batches []batching.Batch) []int {
	counts := make([]int, len(batches))
	for i, batch := range batches {
		counts[i] = batch.Rows
	}
	return counts
}

// assertRow fails t unless got equals want, treating NaN values as equal; the
// label names the row in the failure message.
func assertRow(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if !equalFloats(got, want) {
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

// buildAll plans tasks and builds every batch of the plan, in order, so tests
// can assert on a whole call's batches at once.
func buildAll(tasks []batching.Task, opts batching.Options) []batching.Batch {
	plan := batching.NewPlan(tasks, opts)
	batches := make([]batching.Batch, plan.Len())
	for i := range batches {
		batches[i] = plan.Batch(i)
	}
	return batches
}
