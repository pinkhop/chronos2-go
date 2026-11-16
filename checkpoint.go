package chronos2

import (
	"context"
	"fmt"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/network"
)

// Checkpoint is a Chronos-2 checkpoint (a saved copy of a trained model) in
// the layout Hugging Face publishes, ready for [Load]: its configuration file
// and its weights, typically embedded in the binary with //go:embed.
type Checkpoint struct {
	// Name is reported by [Model.Name], such as "chronos-2-small". It must
	// not be empty.
	Name string

	// Config is the contents of the checkpoint's config.json, which
	// describes the model's dimensions and native percentile levels.
	Config []byte

	// Shards holds the contents of the checkpoint's weight files, in any
	// order. Each is a file in the safetensors format (a JSON header followed
	// by raw arrays), and together they must hold every weight the
	// configuration calls for, each in exactly one file, as 32-bit floats.
	//
	// Load may view the shards' memory directly instead of copying it, so
	// the Model can reference the slices for its whole life. The caller must
	// never modify a shard after passing it to Load: the weights are
	// read-only by convention, and a write would silently change every
	// forecast the Model makes.
	Shards [][]byte
}

// Load decodes a checkpoint into a [Model], checking that every weight array
// the configuration calls for is present with the expected name, element
// type, and shape. On little-endian platforms, weights whose bytes are 4-byte
// aligned are used in place without copying, and the Model treats those bytes
// as read-only; others are copied. Each call builds a separate Model, so model
// packages call Load once per process, typically through [sync.OnceValues]; a
// fine-tuned Chronos-2 checkpoint can ship as its own model package the same
// way.
//
// Errors wrap [ErrInvalidCheckpoint]. Load is safe to call concurrently.
func Load(checkpoint Checkpoint) (*Model, error) {
	if checkpoint.Name == "" {
		return nil, fmt.Errorf("%w: empty name", ErrInvalidCheckpoint)
	}
	config, err := network.ParseConfig(checkpoint.Config)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: config: %w", ErrInvalidCheckpoint, checkpoint.Name, err)
	}
	tensors, err := bindTensors(checkpoint.Shards)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", checkpoint.Name, err)
	}
	net, err := network.New(config, lookup(tensors))
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %w", ErrInvalidCheckpoint, checkpoint.Name, err)
	}
	spec := modelSpec{
		name:               checkpoint.Name,
		contextLength:      config.ContextLength,
		outputPatchSize:    config.PatchSize,
		maxOutputPatches:   config.MaxOutputPatches,
		quantiles:          slices.Clone(config.Quantiles),
		embeddingDimension: config.ModelWidth,
		regToken:           config.UseRegToken,
	}
	return newModel(spec, networkForwarder{network: net}, horizonUnroller(spec)), nil
}

// networkForwarder adapts *network.Network to the root package's forwarder
// interface.
type networkForwarder struct {
	// network runs the forward pass.
	network *network.Network
}

// Forward runs one pass of the network.
func (n networkForwarder) Forward(ctx context.Context, batch network.Batch, options forwardOptions) (network.Output, error) {
	return n.network.Forward(ctx, batch, network.ForwardOptions{Parallelism: options.parallelism, Hidden: options.hidden})
}
