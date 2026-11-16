package chronos2_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"testing/synctest"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// countdownContext cancels itself on its n-th Err call, so a test can cancel
// a call deterministically at a chosen point in its work, and counts how
// often the call checks the context after cancellation.
type countdownContext struct {
	// Context is the wrapped context, canceled by cancel.
	context.Context

	// remaining counts Err calls left before cancellation.
	remaining atomic.Int64

	// cancel cancels the embedded context.
	cancel context.CancelFunc

	// checksAfterCancel counts Err calls that returned an error.
	checksAfterCancel atomic.Int64
}

// newCountdownContext returns a context derived from parent that cancels on
// its n-th Err call.
func newCountdownContext(parent context.Context, n int64) *countdownContext {
	ctx, cancel := context.WithCancel(parent)
	countdown := &countdownContext{Context: ctx, cancel: cancel}
	countdown.remaining.Store(n)
	return countdown
}

// Err cancels the context when the countdown reaches zero and reports the
// embedded context's error.
func (c *countdownContext) Err() error {
	if c.remaining.Add(-1) == 0 {
		c.cancel()
	}
	err := c.Context.Err()
	if err != nil {
		c.checksAfterCancel.Add(1)
	}
	return err
}

// TestModel_Forecast_ContextCanceledBeforeCall_ReturnsCanceled verifies that Model.Forecast returns context.Canceled for a context canceled before the call.
func TestModel_Forecast_ContextCanceledBeforeCall_ReturnsCanceled(t *testing.T) {
	t.Parallel()

	// Given
	model := synthetic(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When
	_, err := model.Forecast(ctx, wave(64, 0), 16)

	// Then
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Forecast() error = %v, want context.Canceled", err)
	}
}

// TestModel_Embed_ContextCanceledBeforeCall_ReturnsCanceled verifies that Model.Embed returns context.Canceled for a context canceled before the call.
func TestModel_Embed_ContextCanceledBeforeCall_ReturnsCanceled(t *testing.T) {
	t.Parallel()

	// Given
	model := synthetic(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When
	_, err := model.Embed(ctx, wave(64, 0))

	// Then
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Embed() error = %v, want context.Canceled", err)
	}
}

// TestModel_Calls_CanceledMidCall_StopAtFirstCheckWithoutLeaks verifies that ForecastBatch, a long-horizon Forecast, and Embed, canceled mid-call, stop at their next check without leaking goroutines.
func TestModel_Calls_CanceledMidCall_StopAtFirstCheckWithoutLeaks(t *testing.T) {
	t.Parallel()

	model := synthetic(t)
	histories := make([][]float64, 40)
	for i := range histories {
		histories[i] = wave(80+i, float64(i))
	}
	tests := midCallCancellations(model, histories)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// synctest fails the test if any goroutine the call started is
			// still blocked when the bubble's work ends.
			synctest.Test(t, func(t *testing.T) {
				// Given a context that cancels partway through the call.
				ctx := newCountdownContext(t.Context(), tt.cancelAt)

				// When
				err := tt.call(ctx)

				// Then
				synctest.Wait()
				if !errors.Is(err, context.Canceled) {
					t.Errorf("error = %v, want context.Canceled", err)
				}
				if checks := ctx.checksAfterCancel.Load(); checks != 1 {
					t.Errorf("context checked %d times after cancellation, want 1 (stop at the first check)", checks)
				}
			})
		})
	}
}

// midCallCancellation is one call canceled partway through its work.
type midCallCancellation struct {
	// name labels the subtest.
	name string

	// cancelAt is the context check at which cancellation happens.
	cancelAt int64

	// call runs the Model method under test with ctx.
	call func(ctx context.Context) error
}

// midCallCancellations returns the calls the mid-call cancellation test
// exercises: a many-chunk ForecastBatch over histories, a long-horizon
// Forecast, and an Embed, each on model.
func midCallCancellations(model *chronos2.Model, histories [][]float64) []midCallCancellation {
	return []midCallCancellation{
		{"ForecastBatch over many chunks", 25, func(ctx context.Context) error {
			_, err := model.ForecastBatch(ctx, histories, 24, chronos2.WithBatchSize(1), chronos2.WithMaxParallelism(4))
			return err
		}},
		{"long-horizon Forecast", 12, func(ctx context.Context) error {
			_, err := model.Forecast(ctx, wave(200, 0), 4000, chronos2.WithLongHorizon(), chronos2.WithMaxParallelism(4))
			return err
		}},
		{"Embed", 3, func(ctx context.Context) error {
			_, err := model.Embed(ctx, wave(400, 0), chronos2.WithMaxParallelism(4))
			return err
		}},
	}
}
