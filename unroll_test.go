package chronos2

import (
	"math"
	"testing"
)

// TestHorizonUnroller_HorizonBeyondOnePass_UnrollsWithBoundParallelism verifies that the horizon unroller, given a horizon beyond one pass, unrolls with the bound parallelism.
func TestHorizonUnroller_HorizonBeyondOnePass_UnrollsWithBoundParallelism(t *testing.T) {
	t.Parallel()

	// Given a model whose long horizons go through internal/horizon.
	forward := &fakeForwarder{}
	model := newModel(testSpec, forward, horizonUnroller(testSpec))

	// When
	forecast, err := model.Forecast(t.Context(), ramp(1, 30), 100, WithLongHorizon(), WithMaxParallelism(2))
	// Then
	if err != nil {
		t.Fatalf("Forecast() error = %v", err)
	}
	if forecast.Horizon() != 100 || len(forward.batches) != 2 {
		t.Errorf("horizon %d after %d passes, want 100 after 2", forecast.Horizon(), len(forward.batches))
	}
	for i, options := range forward.options {
		if options.parallelism != 2 || options.hidden {
			t.Errorf("pass %d options = %+v, want parallelism 2 without hidden states", i, options)
		}
	}
	for step, value := range forecast.Median() {
		if math.IsNaN(value) {
			t.Fatalf("median step %d is NaN", step)
		}
	}
}
