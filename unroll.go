package chronos2

import (
	"context"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/horizon"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// boundForwarder adapts a forwarder to horizon.Forwarder, which takes no
// options, by fixing the parallelism for every pass of one call.
type boundForwarder struct {
	// forward runs each pass.
	forward forwarder

	// parallelism is the goroutine limit for every pass.
	parallelism int
}

// Forward runs one quantile-only pass with the bound parallelism.
func (b boundForwarder) Forward(ctx context.Context, batch network.Batch) (network.Output, error) {
	return b.forward.Forward(ctx, batch, forwardOptions{parallelism: b.parallelism})
}

// horizonUnroller returns the longHorizonForecaster for a model described by
// spec, delegating to internal/horizon. Unrolled contexts grow to the model's
// full context length whatever WithContextLength a call passes, as the
// reference does.
func horizonUnroller(spec modelSpec) longHorizonForecaster {
	config := horizon.Config{
		Levels:           spec.quantiles,
		MaxOutputPatches: spec.maxOutputPatches,
		ContextLength:    spec.contextLength,
	}
	return func(ctx context.Context, forward forwarder, batch batching.Batch, steps, parallelism int) ([]float32, error) {
		return horizon.Forecast(ctx, boundForwarder{forward: forward, parallelism: parallelism}, batch, steps, config)
	}
}
