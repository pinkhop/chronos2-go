// Package horizon runs the engine passes that forecast a batch over a whole
// horizon, unrolling autoregressively when the horizon exceeds one pass.
//
// It reproduces the reference pipeline's _predict_batch: a first pass predicts
// up to the model's maximum output patches, and each later round expands every
// row into nine decile paths, appends each path's decile of the previous
// round's forecast to its context, predicts the next chunk, and reduces the
// paths back to the native quantile levels with a weighted quantile.
package horizon

import (
	"context"
	"errors"
	"fmt"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// outputPatchSize is the number of steps one output patch predicts; the
// network's configuration check guarantees it for every loaded checkpoint.
const outputPatchSize = network.PatchSize

// ErrForward reports that the Forwarder failed during a pass. The returned
// error also wraps the Forwarder's own error.
var ErrForward = errors.New("horizon: forward pass failed")

// ErrInvalidOutput reports that a pass returned an output whose shape does not
// match the batch it was given.
var ErrInvalidOutput = errors.New("horizon: forward pass returned a malformed output")

// Forwarder runs one engine pass over a batch. The root package adapts the
// network, binding its own forward options such as parallelism.
type Forwarder interface {
	// Forward predicts batch.Rows × Q × 16·batch.OutputPatches values. It must
	// not modify or retain batch's slices.
	Forward(ctx context.Context, batch network.Batch) (network.Output, error)
}

// Config describes the model limits a forecast runs under.
type Config struct {
	// Levels are the model's native quantile levels, strictly ascending in
	// (0, 1). They should include the unrolled deciles 0.1 through 0.9, as
	// both models' levels do.
	Levels []float64

	// MaxOutputPatches is the most output patches one pass predicts, at least
	// 1; 64 for both models.
	MaxOutputPatches int

	// ContextLength is the model's full context length, which bounds the
	// growing contexts of unrolled rounds regardless of any shorter window the
	// batch was trimmed to. It is at least 16 × MaxOutputPatches.
	ContextLength int
}

// pass holds one round's prediction for the batch's original rows.
type pass struct {
	// values holds rows × Q × steps predictions, row-major.
	values []float32

	// steps is the number of predicted steps per row, 16 × OutputPatches.
	steps int
}

// Forecast predicts horizon steps for every row of batch and returns
// batch.Rows × len(config.Levels) × horizon values, row-major: row, then
// quantile level, then step. A horizon within config.MaxOutputPatches patches
// takes exactly one pass; a longer one unrolls in rounds of up to
// config.MaxOutputPatches patches, each costing a pass over nine times the
// rows.
//
// batch.Future must hold at least horizon steps per row. Forecast checks ctx
// before every pass and returns an error matching ctx.Err() once it is
// canceled; it returns an error matching ErrForward when f fails and
// ErrInvalidOutput when f returns a malformed output. It panics when horizon
// or config is out of range. It never modifies batch and calls f from the
// calling goroutine only.
func Forecast(ctx context.Context, f Forwarder, batch batching.Batch, horizon int, config Config) ([]float32, error) {
	requireValid(batch, horizon, config)
	result := make([]float32, batch.Rows*len(config.Levels)*horizon)

	// Predict the first chunk over the batch's own rows.
	patches := outputPatches(horizon, config.MaxOutputPatches)
	first := network.Batch{
		Rows:             batch.Rows,
		ContextLength:    batch.ContextLength,
		Context:          batch.Context,
		GroupIDs:         batch.GroupIDs,
		OutputPatches:    patches,
		FutureCovariates: futureWindow(batch, 1, 0, patches*outputPatchSize),
	}
	prediction, err := forward(ctx, f, first, len(config.Levels), 1)
	if err != nil {
		return nil, err
	}
	place(result, prediction, 0, horizon, batch.Rows, len(config.Levels))
	if prediction.steps >= horizon {
		return result, nil
	}

	// Unroll the remaining chunks autoregressively.
	if err := unroll(ctx, f, batch, horizon, config, prediction, result); err != nil {
		return nil, err
	}
	return result, nil
}

// requireValid panics when Forecast's preconditions do not hold; the root
// package guarantees them.
func requireValid(batch batching.Batch, horizon int, config Config) {
	switch {
	case horizon < 1 || batch.FutureLength < horizon:
		panic(fmt.Sprintf("horizon: horizon %d with future length %d", horizon, batch.FutureLength))
	case len(config.Levels) == 0 || config.MaxOutputPatches < 1:
		panic(fmt.Sprintf("horizon: invalid config %+v", config))
	case config.ContextLength < config.MaxOutputPatches*outputPatchSize:
		panic(fmt.Sprintf("horizon: context length %d below one pass", config.ContextLength))
	}
}

// outputPatches returns the patches a pass predicts for remaining steps:
// enough to cover them, capped at maxPatches.
func outputPatches(remaining, maxPatches int) int {
	return min((remaining+outputPatchSize-1)/outputPatchSize, maxPatches)
}

// forward runs one pass after checking ctx, and validates the output's shape
// against batch. number is the 1-based pass number used in error messages.
func forward(ctx context.Context, f Forwarder, batch network.Batch, quantiles, number int) (pass, error) {
	if err := ctx.Err(); err != nil {
		return pass{}, fmt.Errorf("horizon: before pass %d: %w", number, err)
	}
	output, err := f.Forward(ctx, batch)
	if err != nil {
		return pass{}, fmt.Errorf("%w: pass %d: %w", ErrForward, number, err)
	}
	steps := batch.OutputPatches * outputPatchSize
	if output.Rows != batch.Rows || output.Quantiles != quantiles || output.Steps != steps ||
		len(output.Values) != batch.Rows*quantiles*steps {
		return pass{}, fmt.Errorf("%w: pass %d: got %d × %d × %d values, want %d × %d × %d",
			ErrInvalidOutput, number, output.Rows, output.Quantiles, output.Steps, batch.Rows, quantiles, steps)
	}
	return pass{values: output.Values, steps: steps}, nil
}

// futureWindow returns batch's future values for steps [start, start+length)
// of every row, each row repeated copies times in a row, padded with NaN past
// batch.FutureLength. The result has batch.Rows × copies × length values.
func futureWindow(batch batching.Batch, copies, start, length int) []float32 {
	window := nanFloats(batch.Rows * copies * length)
	end := min(start+length, batch.FutureLength)
	if start >= end {
		return window
	}
	for r := range batch.Rows {
		source := batch.Future[r*batch.FutureLength+start : r*batch.FutureLength+end]
		for c := range copies {
			copy(window[(r*copies+c)*length:], source)
		}
	}
	return window
}

// place copies prediction into result at step offset done, truncating it at
// horizon. result holds rows × quantiles × horizon values.
func place(result []float32, prediction pass, done, horizon, rows, quantiles int) {
	count := min(prediction.steps, horizon-done)
	for line := range rows * quantiles {
		source := prediction.values[line*prediction.steps : line*prediction.steps+count]
		copy(result[line*horizon+done:], source)
	}
}
