// Package batching converts a call's validated tasks into engine batches.
//
// NewPlan computes everything that depends on the whole call up front, then
// Plan.Batch builds one batch at a time, so a caller that runs and releases
// batches in turn holds only one batch's buffers at once. It orders each
// task's series into engine rows, encodes categorical covariates, assigns
// group IDs, packs tasks into batches of bounded size, trims rows to the
// context window, and pads rows and future values with NaN.
// It follows the reference implementation's preprocessing, except that it never
// lets a batch exceed the row limit unless a single task does on its own.
//
// The root package validates every input and reports the public errors, so
// NewPlan assumes valid input and panics on a violated precondition.
package batching

import "fmt"

// outputPatchSize is the number of steps in one output patch; future rows are
// padded to a whole number of patches. It equals network.PatchSize, which the
// network's configuration check requires of every checkpoint; this package
// keeps its own copy so it does not depend on the network package.
const outputPatchSize = 16

// Task is one validated task: targets to forecast plus the covariates that
// inform them. Every past series has the same length as the targets, and every
// future series holds exactly Options.Horizon values.
type Task struct {
	// Targets holds the series to forecast, at least one, each with at least
	// one step. NaN marks a missing observation.
	Targets [][]float64

	// PastNumeric maps each numeric covariate name to its history. A name
	// absent from FutureNumeric is past-only. It may be nil.
	PastNumeric map[string][]float64

	// FutureNumeric maps each known-future numeric covariate name to its future
	// values. Every key also appears in PastNumeric. NaN marks an unknown
	// future value. It may be nil.
	FutureNumeric map[string][]float64

	// PastCategorical maps each categorical covariate name to its history. The
	// empty string is the missing category. It may be nil.
	PastCategorical map[string][]string

	// FutureCategorical maps each known-future categorical covariate name to
	// its future values. Every key also appears in PastCategorical. It may be
	// nil.
	FutureCategorical map[string][]string
}

// Options controls how a Plan shapes batches.
type Options struct {
	// Horizon is the number of steps to forecast, at least 1. It may exceed
	// one engine pass; every batch keeps the full future length.
	Horizon int

	// ContextLength is the trim window: each row keeps at most its most recent
	// ContextLength steps. It is at least 1.
	ContextLength int

	// BatchSize is the maximum number of rows per batch, at least 1. Only a
	// single task with more rows than BatchSize forms a larger batch.
	BatchSize int

	// CrossLearning puts every row of every task in group 0. The caller must
	// already have rejected a call with more than BatchSize rows, so the call
	// fits one batch.
	CrossLearning bool
}

// TargetRef locates one task target within a batch, so the caller can route a
// row's forecast back to the task and target it belongs to.
type TargetRef struct {
	// Task is the index of the task in the slice passed to NewPlan.
	Task int

	// Target is the index of the target in that task's Targets.
	Target int

	// Row is the index of the target's row within the batch.
	Row int
}

// Batch is one engine batch, ready to slice into network batches.
type Batch struct {
	// Rows is the number of rows, R.
	Rows int

	// ContextLength is the number of steps per row in Context: the longest
	// trimmed row in this batch.
	ContextLength int

	// Context holds Rows × ContextLength values, row-major, oldest step first.
	// NaN marks a missing value and the left padding of shorter rows.
	Context []float32

	// GroupIDs holds one group per row. Rows of one task share an ID, distinct
	// from every other task's within the batch; under cross-learning every ID
	// is 0.
	GroupIDs []int

	// FutureLength is the number of future steps per row: the horizon rounded
	// up to a whole number of output patches.
	FutureLength int

	// Future holds Rows × FutureLength values, row-major. Known-future
	// covariate rows carry their supplied values; NaN marks every unknown
	// value, including all values of target and past-only covariate rows and
	// the padding beyond the horizon.
	Future []float32

	// Targets locates every target row in the batch, in row order.
	Targets []TargetRef
}

// Plan is a call's tasks divided into engine batches, ready to build one at
// a time. A Plan is immutable after NewPlan returns, so its methods are safe
// for concurrent use.
type Plan struct {
	// tasks holds the call's tasks, shared with the caller.
	tasks []Task

	// opts shapes every batch.
	opts Options

	// categories holds the call-wide ordinal codes of every categorical
	// covariate. Read-only.
	categories ordinalCategories

	// spans lists the tasks of each batch, in order.
	spans []taskSpan
}

// NewPlan divides tasks into engine batches, preserving task order, without
// building any batch. Categorical encoding statistics come from each task's
// full history and, for ordinal codes, from every task in the call, so a
// task's encoding never depends on how the call is chunked or which batches
// are built.
//
// The Plan retains tasks without copying them, so the caller must not modify
// them while it uses the Plan; batching never modifies them. A Plan over no
// tasks has no batches. NewPlan panics when opts holds an out-of-range value.
func NewPlan(tasks []Task, opts Options) Plan {
	requireValidOptions(opts)
	return Plan{
		tasks:      tasks,
		opts:       opts,
		categories: newOrdinalCategories(tasks),
		spans:      chunkTasks(taskRowCounts(tasks), opts.BatchSize),
	}
}

// Len returns the number of batches in the plan.
func (p Plan) Len() int {
	return len(p.spans)
}

// Batch builds batch i, for 0 ≤ i < Len, allocating only that batch's rows
// and matrices. Building a batch again returns an equal, independent copy.
// It panics when i is out of range.
func (p Plan) Batch(i int) Batch {
	span := p.spans[i]
	members := make([]preparedTask, span.end-span.start)
	for offset := range members {
		members[offset] = prepareTask(p.tasks[span.start+offset], p.categories)
	}
	return assembleBatch(members, span.start, p.opts)
}

// requireValidOptions panics when opts violates NewPlan's preconditions; the
// root package validates options before calling NewPlan.
func requireValidOptions(opts Options) {
	if opts.Horizon < 1 || opts.ContextLength < 1 || opts.BatchSize < 1 {
		panic(fmt.Sprintf("batching: invalid options %+v", opts))
	}
}
