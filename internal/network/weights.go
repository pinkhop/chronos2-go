package network

import (
	"errors"
	"fmt"
	"slices"
)

// patchFeatures is the number of input features per patch step: the time
// encoding, the value, and the observation mask.
const patchFeatures = 3

// TensorLookup returns a float32 tensor by its checkpoint name, such as
// "encoder.block.0.layer.0.self_attention.q.weight", with its shape outermost
// first. ok is false when the checkpoint has no such tensor. The network
// retains the returned slices and only reads them.
type TensorLookup func(name string) (data []float32, shape []int, ok bool)

// residualBlockWeights holds one ResidualBlock: out = W_out·ReLU(W_hidden·x +
// b_hidden) + b_out + W_res·x + b_res, with weights stored [out, in].
type residualBlockWeights struct {
	// inputWidth, hiddenWidth, and outputWidth are the block's dimensions.
	inputWidth, hiddenWidth, outputWidth int
	// hiddenWeight and hiddenBias form the first projection.
	hiddenWeight, hiddenBias []float32
	// outputWeight and outputBias project the activations to the output.
	outputWeight, outputBias []float32
	// residualWeight and residualBias form the skip projection.
	residualWeight, residualBias []float32
}

// attentionWeights holds one pre-norm attention sublayer: the RMS norm
// weight and the bias-free query, key, value, and output projections.
type attentionWeights struct {
	// norm is the RMS norm weight, ModelWidth values.
	norm []float32
	// query, key, and value are [Heads·HeadWidth, ModelWidth].
	query, key, value []float32
	// output is [ModelWidth, Heads·HeadWidth].
	output []float32
}

// feedForwardWeights holds one pre-norm feed-forward sublayer.
type feedForwardWeights struct {
	// norm is the RMS norm weight, ModelWidth values.
	norm []float32
	// input is [FeedForwardWidth, ModelWidth].
	input []float32
	// output is [ModelWidth, FeedForwardWidth].
	output []float32
}

// layerWeights holds one encoder layer's three sublayers.
type layerWeights struct {
	// time is the time self-attention sublayer.
	time attentionWeights
	// group is the group self-attention sublayer.
	group attentionWeights
	// feedForward is the feed-forward sublayer.
	feedForward feedForwardWeights
}

// weightLoader fetches tensors by name and shape, collecting every problem
// so New can report them together.
type weightLoader struct {
	// lookup finds tensors in the checkpoint.
	lookup TensorLookup
	// problems collects one error per missing or misshapen tensor.
	problems []error
}

// tensor returns the named tensor, recording a problem and returning nil when
// it is missing or its shape or length disagrees with shape.
func (l *weightLoader) tensor(name string, shape ...int) []float32 {
	data, gotShape, ok := l.lookup(name)
	if !ok {
		l.problems = append(l.problems, fmt.Errorf("%w: missing tensor %q", ErrInvalidWeights, name))
		return nil
	}
	size := 1
	for _, dimension := range shape {
		size *= dimension
	}
	if !slices.Equal(gotShape, shape) || len(data) != size {
		l.problems = append(l.problems, fmt.Errorf("%w: tensor %q has shape %v and %d values, want shape %v",
			ErrInvalidWeights, name, gotShape, len(data), shape))
		return nil
	}
	return data
}

// residualBlock loads the ResidualBlock whose tensors share prefix.
func (l *weightLoader) residualBlock(prefix string, inputWidth, hiddenWidth, outputWidth int) residualBlockWeights {
	return residualBlockWeights{
		inputWidth:     inputWidth,
		hiddenWidth:    hiddenWidth,
		outputWidth:    outputWidth,
		hiddenWeight:   l.tensor(prefix+".hidden_layer.weight", hiddenWidth, inputWidth),
		hiddenBias:     l.tensor(prefix+".hidden_layer.bias", hiddenWidth),
		outputWeight:   l.tensor(prefix+".output_layer.weight", outputWidth, hiddenWidth),
		outputBias:     l.tensor(prefix+".output_layer.bias", outputWidth),
		residualWeight: l.tensor(prefix+".residual_layer.weight", outputWidth, inputWidth),
		residualBias:   l.tensor(prefix+".residual_layer.bias", outputWidth),
	}
}

// attention loads the attention sublayer whose tensors share prefix.
func (l *weightLoader) attention(prefix string, config Config) attentionWeights {
	inner := config.Heads * config.HeadWidth
	return attentionWeights{
		norm:   l.tensor(prefix+".layer_norm.weight", config.ModelWidth),
		query:  l.tensor(prefix+".self_attention.q.weight", inner, config.ModelWidth),
		key:    l.tensor(prefix+".self_attention.k.weight", inner, config.ModelWidth),
		value:  l.tensor(prefix+".self_attention.v.weight", inner, config.ModelWidth),
		output: l.tensor(prefix+".self_attention.o.weight", config.ModelWidth, inner),
	}
}

// layer loads encoder layer index.
func (l *weightLoader) layer(index int, config Config) layerWeights {
	prefix := fmt.Sprintf("encoder.block.%d.layer", index)
	return layerWeights{
		time:  l.attention(prefix+".0", config),
		group: l.attention(prefix+".1", config),
		feedForward: feedForwardWeights{
			norm:   l.tensor(prefix+".2.layer_norm.weight", config.ModelWidth),
			input:  l.tensor(prefix+".2.mlp.wi.weight", config.FeedForwardWidth, config.ModelWidth),
			output: l.tensor(prefix+".2.mlp.wo.weight", config.ModelWidth, config.FeedForwardWidth),
		},
	}
}

// New builds a network from a parsed config and its checkpoint's tensors,
// validating that every tensor the forward pass reads exists with the shape
// the config implies. It retains the tensors' slices without copying them.
// It returns an error wrapping ErrInvalidConfig when config has impossible
// dimensions (ParseConfig never produces one), and otherwise an error
// wrapping ErrInvalidWeights; when several tensors are wrong, the error joins
// one problem per tensor.
func New(config Config, tensors TensorLookup) (*Network, error) {
	if err := validateDimensions(config); err != nil {
		return nil, err
	}
	if tensors == nil {
		return nil, fmt.Errorf("%w: no tensor lookup", ErrInvalidWeights)
	}
	loader := weightLoader{lookup: tensors}
	patchInput := config.PatchSize * patchFeatures
	network := &Network{
		config:      config,
		inputBlock:  loader.residualBlock("input_patch_embedding", patchInput, config.FeedForwardWidth, config.ModelWidth),
		outputBlock: loader.residualBlock("output_patch_embedding", config.ModelWidth, config.FeedForwardWidth, len(config.Quantiles)*config.PatchSize),
		layers:      make([]layerWeights, config.Layers),
		finalNorm:   loader.tensor("encoder.final_layer_norm.weight", config.ModelWidth),
	}
	for index := range config.Layers {
		network.layers[index] = loader.layer(index, config)
	}
	// Only the [REG] token reads the shared embedding; a checkpoint without
	// one stores a single unused entry, so its shape is not checked.
	var shared []float32
	if config.UseRegToken {
		shared = loader.tensor("shared.weight", config.VocabularySize, config.ModelWidth)
	}
	if err := errors.Join(loader.problems...); err != nil {
		return nil, err
	}
	if config.UseRegToken {
		network.regEmbedding = shared[config.RegTokenID*config.ModelWidth : (config.RegTokenID+1)*config.ModelWidth]
	}
	return network, nil
}

// validateDimensions reports, as an error wrapping ErrInvalidConfig, a config
// whose dimensions no checkpoint can have: a non-positive width, count, or
// length, an odd head width (rotary embeddings rotate pairs), no quantiles,
// or a [REG] token ID outside the shared embedding. It returns nil for every
// config ParseConfig produces.
func validateDimensions(config Config) error {
	if config.Layers < 0 || config.ModelWidth <= 0 || config.Heads <= 0 || config.HeadWidth <= 0 ||
		config.HeadWidth%2 != 0 || config.FeedForwardWidth <= 0 || config.PatchSize <= 0 ||
		config.ContextLength <= 0 || config.MaxOutputPatches <= 0 || config.TimeEncodingScale <= 0 ||
		len(config.Quantiles) == 0 {
		return fmt.Errorf("%w: impossible dimensions in %+v", ErrInvalidConfig, config)
	}
	if config.UseRegToken && (config.RegTokenID < 0 || config.RegTokenID >= config.VocabularySize) {
		return fmt.Errorf("%w: [REG] token ID %d outside a vocabulary of %d", ErrInvalidConfig,
			config.RegTokenID, config.VocabularySize)
	}
	return nil
}
