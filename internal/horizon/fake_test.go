package horizon_test

import (
	"context"
	"math"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/horizon"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// nan is the float32 not-a-number value marking missing, padded, and unknown
// values.
var nan = float32(math.NaN())

// smallLevels are the small model's native quantile levels, which include
// every unrolled decile.
var smallLevels = []float64{0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}

// networkBatch abbreviates network.Batch in test closures.
type networkBatch = network.Batch

// valueFunc returns the fake prediction for one row, quantile index, and step
// of a pass; pass counts from 0 in call order.
type valueFunc func(pass, row, quantile, step int) float32

// fakeForwarder is a hand-written Forwarder that records a copy of every
// batch it receives and returns predictions computed by value. It is not safe
// for concurrent use; horizon.Forecast calls it from one goroutine.
type fakeForwarder struct {
	// levels sets the number of quantiles in each output.
	levels []float64

	// value computes each predicted value.
	value valueFunc

	// failPass is the 1-based pass on which Forward returns failure; 0 never
	// fails.
	failPass int

	// failure is the error returned on failPass.
	failure error

	// stepsDelta is added to the Steps of every output, to simulate a
	// malformed output when nonzero.
	stepsDelta int

	// batches records a deep copy of each received batch, in call order.
	batches []network.Batch
}

// Forward records batch and returns a prediction grid of batch.Rows ×
// len(levels) × 16·OutputPatches values, or failure on failPass.
func (f *fakeForwarder) Forward(_ context.Context, batch network.Batch) (network.Output, error) {
	pass := len(f.batches)
	f.batches = append(f.batches, cloneBatch(batch))
	if f.failPass == pass+1 {
		return network.Output{}, f.failure
	}
	quantiles, steps := len(f.levels), batch.OutputPatches*16
	values := make([]float32, batch.Rows*quantiles*steps)
	for r := range batch.Rows {
		for q := range quantiles {
			for s := range steps {
				values[(r*quantiles+q)*steps+s] = f.value(pass, r, q, s)
			}
		}
	}
	return network.Output{Rows: batch.Rows, Quantiles: quantiles, Steps: steps + f.stepsDelta, Values: values}, nil
}

// cloneBatch returns a deep copy of batch, so later mutation by the caller
// cannot change what the fake recorded.
func cloneBatch(batch network.Batch) network.Batch {
	batch.Context = slices.Clone(batch.Context)
	batch.GroupIDs = slices.Clone(batch.GroupIDs)
	batch.FutureCovariates = slices.Clone(batch.FutureCovariates)
	return batch
}

// gridValue is a valueFunc that encodes its coordinates, so a value reveals
// the row, quantile index, and step that produced it.
func gridValue(_, row, quantile, step int) float32 {
	return float32(100*row + 10*quantile + step)
}

// unrollConfig returns a configuration that unrolls every 16 steps: one
// output patch per pass and a 20-step full context window.
func unrollConfig() horizon.Config {
	return horizon.Config{Levels: smallLevels, MaxOutputPatches: 1, ContextLength: 20}
}

// realConfig returns the real models' limits: 64 output patches per pass and
// an 8192-step context window.
func realConfig() horizon.Config {
	return horizon.Config{Levels: smallLevels, MaxOutputPatches: 64, ContextLength: 8192}
}

// newBatch returns a batching.Batch with the given context rows (all the same
// length), group IDs, and future rows (all futureLength long; nil rows are
// all NaN).
func newBatch(contextRows [][]float32, groups []int, futureLength int, futureRows map[int][]float32) batching.Batch {
	rows, length := len(contextRows), len(contextRows[0])
	future := nanFloats(rows * futureLength)
	for r, values := range futureRows {
		copy(future[r*futureLength:], values)
	}
	return batching.Batch{
		Rows:          rows,
		ContextLength: length,
		Context:       slices.Concat(contextRows...),
		GroupIDs:      groups,
		FutureLength:  futureLength,
		Future:        future,
	}
}

// nanFloats returns n NaN values.
func nanFloats(n int) []float32 {
	return slices.Repeat([]float32{nan}, n)
}

// equalFloats reports whether a and b hold the same values in order, treating
// any two NaN values as equal.
func equalFloats(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool {
		return x == y || (x != x && y != y)
	})
}

// field extracts pass-recorded values for summaries in failure messages.
func field[T any](batches []network.Batch, get func(network.Batch) T) []T {
	values := make([]T, len(batches))
	for i, batch := range batches {
		values[i] = get(batch)
	}
	return values
}

// rowOf returns row r of a row-major matrix whose rows are width values long.
func rowOf(values []float32, r, width int) []float32 {
	return values[r*width : (r+1)*width]
}
