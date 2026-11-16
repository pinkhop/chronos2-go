package chronos2

import (
	"fmt"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// boundTensor is one checkpoint weight ready for the network.
type boundTensor struct {
	// data holds the tensor's values, aliasing the shard when possible.
	// Read-only.
	data []float32

	// shape lists the tensor's dimensions, outermost first.
	shape []int
}

// bindTensors parses every shard and returns its float32 tensors by name,
// viewing each tensor's bytes in place when aligned on a little-endian
// platform and decoding a copy otherwise. It returns an error wrapping
// ErrInvalidCheckpoint when a shard fails to parse, a tensor is not F32, or a
// name appears in more than one shard.
func bindTensors(shards [][]byte) (map[string]boundTensor, error) {
	if len(shards) == 0 {
		return nil, fmt.Errorf("%w: no shards", ErrInvalidCheckpoint)
	}
	tensors := make(map[string]boundTensor)
	for shardIndex, shard := range shards {
		file, err := safetensors.Parse(shard)
		if err != nil {
			return nil, fmt.Errorf("%w: shard %d: %w", ErrInvalidCheckpoint, shardIndex, err)
		}
		for name, tensor := range file.Tensors {
			if _, duplicate := tensors[name]; duplicate {
				return nil, fmt.Errorf("%w: tensor %q appears in more than one shard", ErrInvalidCheckpoint, name)
			}
			bound, err := bindTensor(tensor)
			if err != nil {
				return nil, fmt.Errorf("%w: shard %d: %w", ErrInvalidCheckpoint, shardIndex, err)
			}
			tensors[name] = bound
		}
	}
	return tensors, nil
}

// bindTensor converts one parsed tensor, which must be F32, to a boundTensor.
func bindTensor(tensor safetensors.Tensor) (boundTensor, error) {
	if tensor.DType != safetensors.F32 {
		return boundTensor{}, fmt.Errorf("tensor %q is %v, want F32", tensor.Name, tensor.DType)
	}
	if data, ok := aliasFloat32(tensor.Data); ok {
		return boundTensor{data: data, shape: tensor.Shape}, nil
	}
	data, err := tensor.Float32s()
	if err != nil {
		return boundTensor{}, err
	}
	return boundTensor{data: data, shape: tensor.Shape}, nil
}

// lookup returns a network.TensorLookup-shaped function over tensors. The
// shape it returns is a copy; the data is shared and read-only.
func lookup(tensors map[string]boundTensor) func(name string) ([]float32, []int, bool) {
	return func(name string) ([]float32, []int, bool) {
		tensor, ok := tensors[name]
		if !ok {
			return nil, nil, false
		}
		return tensor.data, slices.Clone(tensor.shape), true
	}
}
