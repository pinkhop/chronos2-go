// Package network runs the Chronos-2 forward pass over an engine batch: it
// normalizes and patches each row, embeds the patches, applies the encoder's
// time attention, group attention, and feed-forward sublayers, and decodes
// quantile forecasts. It reproduces the reference implementation
// (chronos-forecasting 2.3.2) in float32, including its unscaled attention
// scores, additive masks, T5-style RMS norms, and rotate-half rotary
// embeddings.
package network

import (
	"context"
	"fmt"
)

// Network is a Chronos-2 model ready to run forward passes. It is immutable
// after New and safe for concurrent use by any number of goroutines.
type Network struct {
	// config holds the architecture hyperparameters.
	config Config
	// inputBlock embeds context and future patches into tokens.
	inputBlock residualBlockWeights
	// outputBlock decodes output tokens into quantile patches.
	outputBlock residualBlockWeights
	// layers holds the encoder layers in order.
	layers []layerWeights
	// finalNorm is the encoder's final RMS norm weight.
	finalNorm []float32
	// regEmbedding is the [REG] token's embedding, or nil without one.
	regEmbedding []float32
}

// ForwardOptions controls one forward pass.
type ForwardOptions struct {
	// Parallelism is the most goroutines the pass may use, including the
	// calling goroutine; values below 1 mean 1.
	Parallelism int
	// Hidden requests the final hidden states in Output.Hidden.
	Hidden bool
}

// StageObserver receives intermediate tensors during ForwardObserved, named
// as in the reference fixtures: loc, scale, normalized_context,
// patched_context, patched_future, input_embeds, layer_00 through layer_NN,
// hidden, quantiles_normalized, and quantiles. It is called on the calling
// goroutine, and values is valid only during the call.
type StageObserver func(name string, values []float32)

// Config returns the network's architecture hyperparameters. The returned
// Quantiles slice is a copy.
func (n *Network) Config() Config {
	config := n.config
	config.Quantiles = append([]float64(nil), n.config.Quantiles...)
	return config
}

// Forward runs one forward pass over batch and returns its quantile
// forecasts in the caller's units, plus the final hidden states when
// options.Hidden is set. It returns an error wrapping ErrInvalidBatch when
// the batch is malformed, or wrapping ctx.Err() when ctx ends before the pass
// completes; ctx is checked between stages and before every sublayer of every
// encoder layer. The batch is only read. Forward allocates its working memory
// per call: one
// Rows × Tokens × ModelWidth float32 residual stream (two with
// options.Hidden), block-sized scratch for the layers' intermediates, and
// per-worker attention scratch that grows with Rows² for group attention.
func (n *Network) Forward(ctx context.Context, batch Batch, options ForwardOptions) (Output, error) {
	return n.ForwardObserved(ctx, batch, options, nil)
}

// ForwardObserved is Forward with a StageObserver that sees every
// intermediate tensor, which lets parity tests localize a divergence to one
// stage. observe may be nil.
func (n *Network) ForwardObserved(ctx context.Context, batch Batch, options ForwardOptions, observe StageObserver) (Output, error) {
	return n.forward(ctx, batch, options, observe, passTuning{blockTokens: defaultBlockTokens})
}

// passTuning holds internal knobs that change how a pass is organized but
// never its results; tests vary them to exercise every code path on small
// inputs.
type passTuning struct {
	// blockTokens is the target number of tokens per block.
	blockTokens int
	// generalGroups disables the singleton-group shortcut in group attention.
	generalGroups bool
}

// forward validates and runs a batch with explicit tuning.
func (n *Network) forward(ctx context.Context, batch Batch, options ForwardOptions, observe StageObserver, tuning passTuning) (Output, error) {
	if err := n.validateBatch(batch); err != nil {
		return Output{}, err
	}
	pass := newForwardPass(n, batch, max(options.Parallelism, 1), observe, tuning)
	return pass.run(ctx, options.Hidden)
}

// validateBatch checks the batch's dimensions against each other and against
// the model's limits.
func (n *Network) validateBatch(batch Batch) error {
	switch {
	case batch.Rows < 1:
		return fmt.Errorf("%w: %d rows", ErrInvalidBatch, batch.Rows)
	case batch.ContextLength < 1 || batch.ContextLength > n.config.ContextLength:
		return fmt.Errorf("%w: context length %d outside [1, %d]", ErrInvalidBatch, batch.ContextLength, n.config.ContextLength)
	case batch.OutputPatches < 1 || batch.OutputPatches > n.config.MaxOutputPatches:
		return fmt.Errorf("%w: %d output patches outside [1, %d]", ErrInvalidBatch, batch.OutputPatches, n.config.MaxOutputPatches)
	case len(batch.Context) != batch.Rows*batch.ContextLength:
		return fmt.Errorf("%w: %d context values, want %d", ErrInvalidBatch, len(batch.Context), batch.Rows*batch.ContextLength)
	case len(batch.GroupIDs) != batch.Rows:
		return fmt.Errorf("%w: %d group IDs, want %d", ErrInvalidBatch, len(batch.GroupIDs), batch.Rows)
	case len(batch.FutureCovariates) != batch.Rows*batch.OutputPatches*n.config.PatchSize:
		return fmt.Errorf("%w: %d future values, want %d", ErrInvalidBatch,
			len(batch.FutureCovariates), batch.Rows*batch.OutputPatches*n.config.PatchSize)
	}
	return nil
}

// layerStageName returns the observer name of encoder layer index's output.
func layerStageName(index int) string {
	return fmt.Sprintf("layer_%02d", index)
}
