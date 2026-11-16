package chronos2

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/batching"
)

// TestModel_ForecastBatch_SmallBatchSize_RoutesEachHistoryToItsForecast verifies that Model.ForecastBatch, given small batch size, routes each history to its forecast.
func TestModel_ForecastBatch_SmallBatchSize_RoutesEachHistoryToItsForecast(t *testing.T) {
	t.Parallel()

	// Given histories ending in 10, 20, and 30, one per engine batch.
	model, forward, _ := newFakeModel()
	histories := [][]float64{ramp(1, 10), ramp(1, 20), ramp(1, 30)}

	// When
	forecasts, err := model.ForecastBatch(t.Context(), histories, 5, WithBatchSize(1))
	// Then
	if err != nil {
		t.Fatalf("ForecastBatch() error = %v", err)
	}
	if len(forward.batches) != 3 {
		t.Errorf("engine passes = %d, want 3", len(forward.batches))
	}
	for i, marker := range []float32{10, 20, 30} {
		if got, want := forecasts[i].lines[6], expectedLine(marker, 6, 5); !slices.Equal(got, want) {
			t.Errorf("forecast %d median = %v, want %v", i, got, want)
		}
	}
}

// TestModel_Forecast_ShortHorizon_RequestsWholePatchesAndTruncates verifies that Model.Forecast, given short horizon, requests whole patches and truncates.
func TestModel_Forecast_ShortHorizon_RequestsWholePatchesAndTruncates(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()

	// When
	forecast, err := model.Forecast(t.Context(), ramp(1, 40), 20)
	// Then
	if err != nil {
		t.Fatalf("Forecast() error = %v", err)
	}
	if batch := forward.batches[0]; batch.OutputPatches != 2 || len(batch.FutureCovariates) != 32 {
		t.Errorf("batch patches = %d, future = %d values; want 2 and 32", batch.OutputPatches, len(batch.FutureCovariates))
	}
	if len(forecast.lines) != len(testLevels) || forecast.Horizon() != 20 {
		t.Errorf("forecast has %d lines of %d steps, want %d of 20", len(forecast.lines), forecast.Horizon(), len(testLevels))
	}
}

// TestModel_ForecastMultivariate_TwoVariates_ShareOneGroup verifies that Model.ForecastMultivariate places two variates in one group.
func TestModel_ForecastMultivariate_TwoVariates_ShareOneGroup(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()

	// When
	forecasts, err := model.ForecastMultivariate(t.Context(), [][]float64{ramp(1, 8), ramp(5, 8)}, 4)
	// Then
	if err != nil {
		t.Fatalf("ForecastMultivariate() error = %v", err)
	}
	if groups := forward.batches[0].GroupIDs; len(groups) != 2 || groups[0] != groups[1] {
		t.Errorf("group IDs = %v, want two equal IDs", groups)
	}
	if got := forecasts[1].lines[0][0]; got != 12 {
		t.Errorf("variate 1 first value = %v, want 12", got)
	}
}

// TestModel_ForecastWithCovariates_KnownFuture_PassesFutureValues verifies that Model.ForecastWithCovariates, given known future, passes future values.
func TestModel_ForecastWithCovariates_KnownFuture_PassesFutureValues(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()
	task := CovariateTask{
		Target:     ramp(1, 6),
		Covariates: Covariates{Past: map[string][]float64{"x": ramp(0, 6)}, Future: map[string][]float64{"x": {7, 8, 9}}},
	}

	// When
	_, err := model.ForecastWithCovariates(t.Context(), task, 3)
	// Then
	if err != nil {
		t.Fatalf("ForecastWithCovariates() error = %v", err)
	}
	future := forward.batches[0].FutureCovariates
	if len(future) != 32 || future[16] != 7 || future[18] != 9 || !math.IsNaN(float64(future[19])) || !math.IsNaN(float64(future[0])) {
		t.Errorf("future covariates = %v, want a NaN target row then 7 8 9 and NaN padding", future)
	}
}

// TestModel_Forecast_LongHorizonOption_UsesUnroll verifies that Model.Forecast, given long horizon option, uses unroll.
func TestModel_Forecast_LongHorizonOption_UsesUnroll(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, unroll := newFakeModel()

	// When
	forecast, err := model.Forecast(t.Context(), ramp(1, 30), 100, WithLongHorizon())
	// Then
	if err != nil {
		t.Fatalf("Forecast() error = %v", err)
	}
	if len(forward.batches) != 0 || !slices.Equal(unroll.horizons, []int{100}) {
		t.Errorf("passes = %d, unroll horizons = %v; want 0 and [100]", len(forward.batches), unroll.horizons)
	}
	if forecast.Horizon() != 100 {
		t.Errorf("Horizon() = %d, want 100", forecast.Horizon())
	}
}

// TestModel_Forecast_MaxParallelism_PassedToEngine verifies that Model.Forecast passes WithMaxParallelism to the engine.
func TestModel_Forecast_MaxParallelism_PassedToEngine(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 30), 4, WithMaxParallelism(3))
	// Then
	if err != nil {
		t.Fatalf("Forecast() error = %v", err)
	}
	if got := forward.options[0]; got.parallelism != 3 || got.hidden {
		t.Errorf("options = %+v, want parallelism 3 without hidden states", got)
	}
}

// TestModel_Forecast_CanceledContext_ReturnsContextError verifies that Model.Forecast, given canceled context, returns context error.
func TestModel_Forecast_CanceledContext_ReturnsContextError(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When
	_, err := model.Forecast(ctx, ramp(1, 30), 4)

	// Then
	if !errors.Is(err, context.Canceled) || len(forward.batches) != 0 {
		t.Errorf("error = %v after %d passes, want context.Canceled before any pass", err, len(forward.batches))
	}
}

// TestModel_Forecast_EngineFailure_ReturnsInvalidCheckpoint verifies that Model.Forecast, given engine failure, returns invalid checkpoint.
func TestModel_Forecast_EngineFailure_ReturnsInvalidCheckpoint(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()
	forward.err = errors.New("tensor shape mismatch")

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 30), 4)

	// Then
	if !errors.Is(err, ErrInvalidCheckpoint) {
		t.Errorf("error = %v, want ErrInvalidCheckpoint", err)
	}
}

// TestModel_Forecast_EngineCanceled_ReturnsContextError verifies that Model.Forecast, given a canceled engine pass, returns the context's error.
func TestModel_Forecast_EngineCanceled_ReturnsContextError(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()
	forward.err = context.DeadlineExceeded

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 30), 4)

	// Then
	if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrInvalidCheckpoint) {
		t.Errorf("error = %v, want only context.DeadlineExceeded", err)
	}
}

// TestModel_Forecast_WrongOutputShape_ReturnsInvalidCheckpoint verifies that Model.Forecast, given wrong output shape, returns invalid checkpoint.
func TestModel_Forecast_WrongOutputShape_ReturnsInvalidCheckpoint(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()
	forward.extraRows = 1

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 30), 4)

	// Then
	if !errors.Is(err, ErrInvalidCheckpoint) {
		t.Errorf("error = %v, want ErrInvalidCheckpoint", err)
	}
}

// TestModel_ForecastTasks_NoTasks_ReturnsEmptyResult verifies that Model.ForecastTasks, given no tasks, returns empty result.
func TestModel_ForecastTasks_NoTasks_ReturnsEmptyResult(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()

	// When
	results, err := model.ForecastTasks(t.Context(), nil, 4)
	// Then
	if err != nil || len(results) != 0 || len(forward.batches) != 0 {
		t.Errorf("results = %v, error = %v, passes = %d; want empty, nil, 0", results, err, len(forward.batches))
	}
}

// TestModel_Forecast_UnrollReturnsWrongLength_ReturnsInvalidCheckpoint
// verifies that a long-horizon result of the wrong size is reported as
// ErrInvalidCheckpoint instead of panicking while routing values to targets.
func TestModel_Forecast_UnrollReturnsWrongLength_ReturnsInvalidCheckpoint(t *testing.T) {
	t.Parallel()

	// Given
	forward := &fakeForwarder{}
	short := func(context.Context, forwarder, batching.Batch, int, int) ([]float32, error) {
		return make([]float32, 3), nil
	}
	model := newModel(testSpec, forward, short)

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 30), 100, WithLongHorizon())

	// Then
	if !errors.Is(err, ErrInvalidCheckpoint) {
		t.Errorf("Forecast() error = %v, want ErrInvalidCheckpoint", err)
	}
}

// TestModel_ForecastBatch_CanceledBetweenBatches_StopsBeforeNextPass verifies
// that a call spanning several engine batches checks its context between
// them, so canceling during one pass prevents the next.
func TestModel_ForecastBatch_CanceledBetweenBatches_StopsBeforeNextPass(t *testing.T) {
	t.Parallel()

	// Given three histories, one per batch, and a context canceled as soon as
	// the first pass finishes
	model, forward, _ := newFakeModel()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	forward.afterForward = cancel
	histories := [][]float64{ramp(1, 10), ramp(1, 20), ramp(1, 30)}

	// When
	_, err := model.ForecastBatch(ctx, histories, 5, WithBatchSize(1))

	// Then
	if !errors.Is(err, context.Canceled) || len(forward.batches) != 1 {
		t.Errorf("error = %v after %d passes, want context.Canceled after 1", err, len(forward.batches))
	}
}
