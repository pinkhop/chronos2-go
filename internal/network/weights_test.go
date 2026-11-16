package network

import (
	"errors"
	"testing"
)

// lookupWith returns base with the tensor name replaced by a zero tensor of
// shape, so a test can present a checkpoint that differs in one tensor.
func lookupWith(base TensorLookup, name string, shape ...int) TensorLookup {
	size := 1
	for _, dimension := range shape {
		size *= dimension
	}
	replacement := make([]float32, size)
	return func(requested string) ([]float32, []int, bool) {
		if requested == name {
			return replacement, shape, true
		}
		return base(requested)
	}
}

// noRegTokenNetwork builds the synthetic network as a checkpoint without a
// [REG] token: the config ParseConfig produces for one, and a one-entry
// shared embedding as the reference stores it.
func noRegTokenNetwork(t *testing.T) *Network {
	t.Helper()
	config := syntheticConfig(t)
	config.UseRegToken, config.VocabularySize, config.RegTokenID = false, 1, 0
	lookup := lookupWith(syntheticLookup(t), "shared.weight", 1, config.ModelWidth)
	network, err := New(config, lookup)
	if err != nil {
		t.Fatalf("New without a [REG] token: %v", err)
	}
	return network
}

// TestNew_NoRegToken_BuildsWithReferenceSharedEmbedding verifies that New
// accepts a checkpoint without a [REG] token, whose shared embedding the
// reference stores with one entry and the forward pass never reads.
func TestNew_NoRegToken_BuildsWithReferenceSharedEmbedding(t *testing.T) {
	t.Parallel()

	// Given the config and tensors of a checkpoint without a [REG] token
	config := syntheticConfig(t)
	config.UseRegToken, config.VocabularySize, config.RegTokenID = false, 1, 0
	lookup := lookupWith(syntheticLookup(t), "shared.weight", 1, config.ModelWidth)

	// When the network is built
	_, err := New(config, lookup)
	// Then it builds
	if err != nil {
		t.Errorf("New: %v", err)
	}
}

// TestForward_NoRegToken_RowsHoldContextThenOutputTokens verifies that a
// network without a [REG] token lays out each row as its context patches
// followed directly by its output patches.
func TestForward_NoRegToken_RowsHoldContextThenOutputTokens(t *testing.T) {
	t.Parallel()

	// Given a network without a [REG] token and a 40-step, one-patch batch
	network := noRegTokenNetwork(t)
	batch := singleRowBatch(40)

	// When the batch is forwarded with hidden states
	output, err := network.Forward(t.Context(), batch, ForwardOptions{Parallelism: 2, Hidden: true})
	// Then each row holds three context tokens and one output token
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Tokens != 4 || len(output.Hidden) != 4*output.Width {
		t.Errorf("Tokens = %d, len(Hidden) = %d, want 4 tokens of width %d", output.Tokens, len(output.Hidden), output.Width)
	}
}

// TestNew_RegTokenIDOutsideVocabulary_ReturnsErrInvalidConfig verifies that
// New rejects a [REG] token ID the shared embedding cannot hold instead of
// panicking.
func TestNew_RegTokenIDOutsideVocabulary_ReturnsErrInvalidConfig(t *testing.T) {
	t.Parallel()

	for _, id := range []int{-1, 2} {
		t.Run("", func(t *testing.T) {
			t.Parallel()

			// Given a [REG] token ID outside the two-entry vocabulary
			config := syntheticConfig(t)
			config.RegTokenID = id

			// When the network is built
			_, err := New(config, syntheticLookup(t))

			// Then the config is rejected
			if !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("RegTokenID %d: err = %v, want ErrInvalidConfig", id, err)
			}
		})
	}
}

// TestNew_NonPositiveDimension_ReturnsErrInvalidConfig verifies that New
// reports a non-positive dimension as an invalid config rather than as
// mismatched weights or a network that rejects every batch.
func TestNew_NonPositiveDimension_ReturnsErrInvalidConfig(t *testing.T) {
	t.Parallel()

	cases := map[string]func(config *Config){
		"heads":              func(config *Config) { config.Heads = 0 },
		"feed-forward width": func(config *Config) { config.FeedForwardWidth = 0 },
		"context length":     func(config *Config) { config.ContextLength = 0 },
		"max output patches": func(config *Config) { config.MaxOutputPatches = 0 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Given a config with one non-positive dimension
			config := syntheticConfig(t)
			mutate(&config)

			// When the network is built
			_, err := New(config, syntheticLookup(t))

			// Then the config is rejected
			if !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("err = %v, want ErrInvalidConfig", err)
			}
		})
	}
}
