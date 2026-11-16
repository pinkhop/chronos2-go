package horizon_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/horizon"
)

// TestForecast_HorizonWithinOnePass_RunsOnePassOverBatchAsIs verifies that Forecast, given horizon within one pass, runs one pass over batch as is.
func TestForecast_HorizonWithinOnePass_RunsOnePassOverBatchAsIs(t *testing.T) {
	t.Parallel()

	// Given a two-row batch, a 20-step horizon, and a known future row
	known := []float32{1, nan, 3}
	batch := newBatch([][]float32{{1, 2, 3}, {nan, 5, 6}}, []int{0, 1}, 32, map[int][]float32{1: known})
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 20, realConfig())
	// Then one two-patch pass receives the batch's context, groups, and future
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	if len(fake.batches) != 1 {
		t.Fatalf("passes = %d, want 1", len(fake.batches))
	}
	got := fake.batches[0]
	if got.OutputPatches != 2 || got.Rows != 2 || got.ContextLength != 3 {
		t.Errorf("OutputPatches, Rows, ContextLength = %d, %d, %d, want 2, 2, 3", got.OutputPatches, got.Rows, got.ContextLength)
	}
	if !equalFloats(got.Context, batch.Context) || !slices.Equal(got.GroupIDs, batch.GroupIDs) {
		t.Errorf("Context, GroupIDs = %v, %v, want %v, %v", got.Context, got.GroupIDs, batch.Context, batch.GroupIDs)
	}
	if !equalFloats(got.FutureCovariates, batch.Future) {
		t.Errorf("FutureCovariates = %v, want %v", got.FutureCovariates, batch.Future)
	}
}

// TestForecast_HorizonWithinOnePass_ReturnsPredictionTruncatedToHorizon verifies that Forecast, given horizon within one pass, returns prediction truncated to horizon.
func TestForecast_HorizonWithinOnePass_ReturnsPredictionTruncatedToHorizon(t *testing.T) {
	t.Parallel()

	// Given a two-row batch and a 20-step horizon predicted in two patches
	batch := newBatch([][]float32{{1}, {2}}, []int{0, 1}, 32, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	result, err := horizon.Forecast(t.Context(), fake, batch, 20, realConfig())
	// Then the result holds Rows × Q × 20 values, row then quantile then step
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	quantiles := len(smallLevels)
	if len(result) != 2*quantiles*20 {
		t.Fatalf("len(result) = %d, want %d", len(result), 2*quantiles*20)
	}
	for r := range 2 {
		for q := range quantiles {
			want := make([]float32, 20)
			for s := range want {
				want[s] = gridValue(0, r, q, s)
			}
			if got := rowOf(result, r*quantiles+q, 20); !equalFloats(got, want) {
				t.Errorf("row %d quantile %d = %v, want %v", r, q, got, want)
			}
		}
	}
}

// TestForecast_HorizonOfMaxPatches_RunsOnePass verifies that Forecast, given horizon of max patches, runs one pass.
func TestForecast_HorizonOfMaxPatches_RunsOnePass(t *testing.T) {
	t.Parallel()

	// Given a horizon of exactly 64 patches
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 1024, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 1024, realConfig())
	// Then a single 64-patch pass covers it
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.OutputPatches }); !slices.Equal(got, []int{64}) {
		t.Errorf("OutputPatches per pass = %v, want [64]", got)
	}
}

// TestForecast_HorizonOneStepBeyondMaxPatches_UnrollsOneOnePatchRound verifies that Forecast unrolls one one-patch round for a horizon one step beyond the maximum.
func TestForecast_HorizonOneStepBeyondMaxPatches_UnrollsOneOnePatchRound(t *testing.T) {
	t.Parallel()

	// Given a horizon one step beyond 64 patches
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 1040, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 1025, realConfig())
	// Then a 64-patch pass is followed by a one-patch pass over nine paths
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.OutputPatches }); !slices.Equal(got, []int{64, 1}) {
		t.Errorf("OutputPatches per pass = %v, want [64 1]", got)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.Rows }); !slices.Equal(got, []int{1, 9}) {
		t.Errorf("Rows per pass = %v, want [1 9]", got)
	}
}

// TestForecast_ForwarderFails_ReturnsWrappedError verifies that Forecast, given forwarder fails, returns wrapped error.
func TestForecast_ForwarderFails_ReturnsWrappedError(t *testing.T) {
	t.Parallel()

	// Given a forwarder that fails on the second pass
	failure := errors.New("engine exploded")
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 32, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue, failPass: 2, failure: failure}

	// When a horizon needing two passes is forecast
	result, err := horizon.Forecast(t.Context(), fake, batch, 32, unrollConfig())

	// Then the error matches both ErrForward and the forwarder's error
	if !errors.Is(err, horizon.ErrForward) || !errors.Is(err, failure) {
		t.Errorf("err = %v, want ErrForward wrapping %v", err, failure)
	}
	if result != nil {
		t.Errorf("result = %v, want nil", result)
	}
}

// TestForecast_CanceledContext_ReturnsContextErrorWithoutForwarding verifies that Forecast, given canceled context, returns context error without forwarding.
func TestForecast_CanceledContext_ReturnsContextErrorWithoutForwarding(t *testing.T) {
	t.Parallel()

	// Given a canceled context
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 16, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(ctx, fake, batch, 16, realConfig())

	// Then the context's error is returned and no pass runs
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if len(fake.batches) != 0 {
		t.Errorf("passes = %d, want 0", len(fake.batches))
	}
}

// TestForecast_MalformedOutput_ReturnsErrInvalidOutput verifies that Forecast, given a malformed output, returns ErrInvalidOutput.
func TestForecast_MalformedOutput_ReturnsErrInvalidOutput(t *testing.T) {
	t.Parallel()

	// Given a forwarder whose outputs claim one step too many
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 16, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue, stepsDelta: 1}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 16, realConfig())

	// Then the malformed output is reported
	if !errors.Is(err, horizon.ErrInvalidOutput) {
		t.Errorf("err = %v, want ErrInvalidOutput", err)
	}
}
