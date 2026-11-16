package chronos2

import (
	"context"
	"errors"
	"fmt"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// forecastTasks is the single route from every forecasting method to the
// engine: it resolves options, validates the input, plans the batches, runs
// them one at a time, and routes every target row's predictions back to its
// task and target. Each batch is built just before its pass and released
// after it, so peak memory follows the batch size rather than the whole call.
// The result holds one slice per task with one Forecast per target.
func (m *Model) forecastTasks(ctx context.Context, tasks []Task, horizon int, options []ForecastOption, label targetLabeler) ([][]Forecast, error) {
	settings, err := collectForecastOptions(options).resolve(m.spec.contextLength)
	if err != nil {
		return nil, err
	}
	if err := checkHorizon(horizon, m.MaxHorizon(), settings.longHorizon); err != nil {
		return nil, err
	}
	if err := validateTasks(tasks, horizon, label); err != nil {
		return nil, err
	}
	if settings.crossLearning {
		if err := checkCrossLearningFits(tasks, settings.batchSize); err != nil {
			return nil, err
		}
	}

	// Plan the batches, then build, run, and release each in turn.
	plan := batching.NewPlan(toBatchingTasks(tasks), batching.Options{
		Horizon:       horizon,
		ContextLength: settings.contextLength,
		BatchSize:     settings.batchSize,
		CrossLearning: settings.crossLearning,
	})
	results := make([][]Forecast, len(tasks))
	for i, task := range tasks {
		results[i] = make([]Forecast, len(task.Targets))
	}
	for i := range plan.Len() {
		if err := m.runBatch(ctx, plan.Batch(i), horizon, settings.parallelism, results); err != nil {
			return nil, err
		}
	}
	return results, nil
}

// runBatch runs one engine batch and stores a Forecast for each of its target
// rows in results, indexed by task and target. predict checks ctx before the
// pass, so a canceled call stops between batches.
func (m *Model) runBatch(ctx context.Context, batch batching.Batch, horizon, parallelism int, results [][]Forecast) error {
	values, err := m.predict(ctx, batch, horizon, parallelism)
	if err != nil {
		return err
	}
	perRow := len(m.spec.quantiles) * horizon
	for _, ref := range batch.Targets {
		results[ref.Task][ref.Target] = newForecast(m.spec.quantiles, values[ref.Row*perRow:(ref.Row+1)*perRow], horizon)
	}
	return nil
}

// toBatchingTasks converts validated public tasks to batching tasks. The
// slices and maps are shared, not copied: batching never modifies them, and
// the call's Plan holds them only until the call returns.
func toBatchingTasks(tasks []Task) []batching.Task {
	converted := make([]batching.Task, len(tasks))
	for i, task := range tasks {
		converted[i] = batching.Task{
			Targets:           task.Targets,
			PastNumeric:       task.Past,
			FutureNumeric:     task.Future,
			PastCategorical:   task.PastCategorical,
			FutureCategorical: task.FutureCategorical,
		}
	}
	return converted
}

// predict returns Rows × Quantiles × horizon predictions for batch, using one
// engine pass when the horizon fits and the long-horizon unroll otherwise.
func (m *Model) predict(ctx context.Context, batch batching.Batch, horizon, parallelism int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("chronos2: forecast canceled: %w", err)
	}
	if horizon > m.MaxHorizon() {
		values, err := m.unroll(ctx, m.forward, batch, horizon, parallelism)
		if err != nil {
			return nil, wrapEngineError(err)
		}
		if want := batch.Rows * len(m.spec.quantiles) * horizon; len(values) != want {
			return nil, fmt.Errorf("%w: long-horizon unroll returned %d values, want %d",
				ErrInvalidCheckpoint, len(values), want)
		}
		return values, nil
	}
	patches := (horizon + m.spec.outputPatchSize - 1) / m.spec.outputPatchSize
	output, err := m.forward.Forward(ctx, network.Batch{
		Rows:             batch.Rows,
		ContextLength:    batch.ContextLength,
		Context:          batch.Context,
		GroupIDs:         batch.GroupIDs,
		OutputPatches:    patches,
		FutureCovariates: futureWindow(batch, patches*m.spec.outputPatchSize),
	}, forwardOptions{parallelism: parallelism})
	if err != nil {
		return nil, wrapEngineError(err)
	}
	if err := m.checkOutput(output, batch.Rows, patches); err != nil {
		return nil, err
	}
	return truncateSteps(output, horizon), nil
}

// futureWindow returns the first steps future values of every row of batch,
// Rows × steps, sharing batch.Future when it already has that width.
func futureWindow(batch batching.Batch, steps int) []float32 {
	if batch.FutureLength == steps {
		return batch.Future
	}
	window := make([]float32, 0, batch.Rows*steps)
	for row := range batch.Rows {
		start := row * batch.FutureLength
		window = append(window, batch.Future[start:start+steps]...)
	}
	return window
}

// truncateSteps keeps the first horizon steps of every row and quantile of
// output, in a new Rows × Quantiles × horizon slice.
func truncateSteps(output network.Output, horizon int) []float32 {
	truncated := make([]float32, 0, output.Rows*output.Quantiles*horizon)
	for line := range output.Rows * output.Quantiles {
		start := line * output.Steps
		truncated = append(truncated, output.Values[start:start+horizon]...)
	}
	return truncated
}

// checkOutput returns an error wrapping ErrInvalidCheckpoint when output's
// shape disagrees with the request, which would otherwise route values to
// the wrong targets or panic.
func (m *Model) checkOutput(output network.Output, rows, patches int) error {
	steps := patches * m.spec.outputPatchSize
	quantiles := len(m.spec.quantiles)
	if output.Rows != rows || output.Quantiles != quantiles || output.Steps != steps || len(output.Values) != rows*quantiles*steps {
		return fmt.Errorf("%w: network returned %d×%d×%d values (%d), want %d×%d×%d",
			ErrInvalidCheckpoint, output.Rows, output.Quantiles, output.Steps, len(output.Values), rows, quantiles, steps)
	}
	return nil
}

// wrapEngineError passes a cancellation through so it matches ctx.Err() and
// otherwise wraps err with ErrInvalidCheckpoint, since the engine fails only
// when the checkpoint and its configuration disagree.
func wrapEngineError(err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("chronos2: forecast canceled: %w", err)
	}
	return fmt.Errorf("%w: engine failure: %w", ErrInvalidCheckpoint, err)
}
