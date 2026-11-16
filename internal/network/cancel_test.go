package network

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
)

// countdownContext cancels itself on its n-th Err call, so a test can cancel
// a forward pass deterministically at a chosen cancellation check.
type countdownContext struct {
	// Context is the wrapped context, canceled by cancel.
	context.Context

	// mutex guards remaining.
	mutex sync.Mutex

	// remaining counts Err calls left before cancellation.
	remaining int

	// cancel cancels the embedded context.
	cancel context.CancelFunc
}

// newCountdownContext returns a context derived from parent that cancels on
// its n-th Err call.
func newCountdownContext(parent context.Context, n int) *countdownContext {
	ctx, cancel := context.WithCancel(parent)
	return &countdownContext{Context: ctx, remaining: n, cancel: cancel}
}

// Err cancels the context when the countdown reaches zero and reports the
// embedded context's error.
func (c *countdownContext) Err() error {
	c.mutex.Lock()
	c.remaining--
	if c.remaining == 0 {
		c.cancel()
	}
	c.mutex.Unlock()
	return c.Context.Err()
}

// TestForward_CanceledInsideFirstLayer_StopsBeforeLayerCompletes verifies that
// Forward checks for cancellation between the sublayers of an encoder layer,
// so a long pass stops within a sublayer of cancellation rather than finishing
// the layer.
func TestForward_CanceledInsideFirstLayer_StopsBeforeLayerCompletes(t *testing.T) {
	t.Parallel()

	// Given a context that cancels on its third check: the first precedes the
	// pass, the second the first layer's time attention, and the third falls
	// inside that layer
	network := syntheticNetwork(t)
	ctx := newCountdownContext(t.Context(), 3)
	defer ctx.cancel()
	var stages []string
	observe := func(name string, _ []float32) { stages = append(stages, name) }

	// When a pass runs
	_, err := network.ForwardObserved(ctx, singleRowBatch(20), ForwardOptions{Parallelism: 1}, observe)

	// Then it returns the cancellation without completing the first layer
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if slices.Contains(stages, layerStageName(0)) {
		t.Errorf("stages = %v, want no %s", stages, layerStageName(0))
	}
}
