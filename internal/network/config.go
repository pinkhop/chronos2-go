package network

import (
	"encoding/json"
	"fmt"
	"slices"
)

// supportedActivation is the only feed-forward and residual-block activation
// Chronos-2 checkpoints use.
const supportedActivation = "relu"

// PatchSize is the number of steps in every input and output patch. Every
// published Chronos-2 checkpoint uses 16, and ParseConfig rejects any other
// size, so packages that shape batches for the network may rely on it.
const PatchSize = 16

// Config holds the architecture hyperparameters of a Chronos-2 checkpoint,
// parsed from its config.json. A Config is a value; treat Quantiles as
// read-only.
type Config struct {
	// ModelWidth is the token width, d_model.
	ModelWidth int
	// FeedForwardWidth is the hidden width of the feed-forward networks and
	// residual blocks, d_ff.
	FeedForwardWidth int
	// HeadWidth is the width of one attention head, d_kv; it is even.
	HeadWidth int
	// Heads is the number of attention heads, num_heads.
	Heads int
	// Layers is the number of encoder layers, num_layers.
	Layers int
	// LayerNormEpsilon is the RMS norm variance floor, layer_norm_epsilon.
	LayerNormEpsilon float32
	// RopeTheta is the rotary embedding base, rope_theta.
	RopeTheta float64
	// VocabularySize is the number of rows of the shared token embedding: 2
	// with a [REG] token, 1 without.
	VocabularySize int
	// RegTokenID is the shared-embedding row of the [REG] token.
	RegTokenID int
	// ContextLength is the longest context the model reads, in steps.
	ContextLength int
	// PatchSize is the number of steps per input and output patch.
	PatchSize int
	// MaxOutputPatches is the most output patches one forward pass predicts.
	MaxOutputPatches int
	// Quantiles lists the predicted quantile levels, strictly ascending in
	// (0, 1).
	Quantiles []float64
	// TimeEncodingScale divides step offsets in the time encoding.
	TimeEncodingScale int
	// UseArcsinh reports whether normalized values pass through asinh.
	UseArcsinh bool
	// UseRegToken reports whether a [REG] token follows the context tokens.
	UseRegToken bool
}

// configDocument mirrors the fields of config.json this engine reads.
type configDocument struct {
	// ModelWidth is d_model.
	ModelWidth int `json:"d_model"`
	// FeedForwardWidth is d_ff.
	FeedForwardWidth int `json:"d_ff"`
	// HeadWidth is d_kv.
	HeadWidth int `json:"d_kv"`
	// Heads is num_heads.
	Heads int `json:"num_heads"`
	// Layers is num_layers.
	Layers int `json:"num_layers"`
	// LayerNormEpsilon is layer_norm_epsilon.
	LayerNormEpsilon float64 `json:"layer_norm_epsilon"`
	// RopeTheta is rope_theta.
	RopeTheta float64 `json:"rope_theta"`
	// Activation is dense_act_fn.
	Activation string `json:"dense_act_fn"`
	// GatedActivation is is_gated_act.
	GatedActivation bool `json:"is_gated_act"`
	// Chronos is the nested chronos_config object; nil when absent.
	Chronos *chronosDocument `json:"chronos_config"`
}

// chronosDocument mirrors config.json's chronos_config object.
type chronosDocument struct {
	// ContextLength is context_length.
	ContextLength int `json:"context_length"`
	// InputPatchSize is input_patch_size.
	InputPatchSize int `json:"input_patch_size"`
	// InputPatchStride is input_patch_stride.
	InputPatchStride int `json:"input_patch_stride"`
	// OutputPatchSize is output_patch_size.
	OutputPatchSize int `json:"output_patch_size"`
	// MaxOutputPatches is max_output_patches.
	MaxOutputPatches int `json:"max_output_patches"`
	// Quantiles is quantiles.
	Quantiles []float64 `json:"quantiles"`
	// TimeEncodingScale is time_encoding_scale; nil means context_length.
	TimeEncodingScale *int `json:"time_encoding_scale"`
	// UseArcsinh is use_arcsinh.
	UseArcsinh bool `json:"use_arcsinh"`
	// UseRegToken is use_reg_token.
	UseRegToken bool `json:"use_reg_token"`
}

// ParseConfig parses a Chronos-2 config.json and validates that it describes
// an architecture this engine supports. Like the reference, it fixes the
// [REG] token at row 1 of a two-row vocabulary when use_reg_token is set,
// uses a one-row vocabulary otherwise, and defaults time_encoding_scale to
// context_length. Every error wraps ErrInvalidConfig.
func ParseConfig(configJSON []byte) (Config, error) {
	var document configDocument
	if err := json.Unmarshal(configJSON, &document); err != nil {
		return Config{}, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	if document.Chronos == nil {
		return Config{}, fmt.Errorf("%w: missing chronos_config", ErrInvalidConfig)
	}
	if err := validateDocument(document); err != nil {
		return Config{}, err
	}
	chronos := document.Chronos
	config := Config{
		ModelWidth:        document.ModelWidth,
		FeedForwardWidth:  document.FeedForwardWidth,
		HeadWidth:         document.HeadWidth,
		Heads:             document.Heads,
		Layers:            document.Layers,
		LayerNormEpsilon:  float32(document.LayerNormEpsilon),
		RopeTheta:         document.RopeTheta,
		VocabularySize:    1,
		ContextLength:     chronos.ContextLength,
		PatchSize:         chronos.InputPatchSize,
		MaxOutputPatches:  chronos.MaxOutputPatches,
		Quantiles:         slices.Clone(chronos.Quantiles),
		TimeEncodingScale: chronos.ContextLength,
		UseArcsinh:        chronos.UseArcsinh,
		UseRegToken:       chronos.UseRegToken,
	}
	if chronos.TimeEncodingScale != nil {
		config.TimeEncodingScale = *chronos.TimeEncodingScale
	}
	if config.UseRegToken {
		config.VocabularySize, config.RegTokenID = 2, 1
	}
	if config.TimeEncodingScale <= 0 {
		return Config{}, fmt.Errorf("%w: time_encoding_scale %d is not positive", ErrInvalidConfig, config.TimeEncodingScale)
	}
	return config, nil
}

// validateDocument checks the parsed document's dimensions, activation,
// patching, and quantile levels.
func validateDocument(document configDocument) error {
	chronos := document.Chronos
	positive := []struct {
		name  string
		value int
	}{
		{"d_model", document.ModelWidth},
		{"d_ff", document.FeedForwardWidth},
		{"d_kv", document.HeadWidth},
		{"num_heads", document.Heads},
		{"num_layers", document.Layers},
		{"context_length", chronos.ContextLength},
		{"input_patch_size", chronos.InputPatchSize},
		{"max_output_patches", chronos.MaxOutputPatches},
	}
	for _, field := range positive {
		if field.value <= 0 {
			return fmt.Errorf("%w: %s %d is not positive", ErrInvalidConfig, field.name, field.value)
		}
	}
	switch {
	case document.HeadWidth%2 != 0:
		return fmt.Errorf("%w: d_kv %d is odd; rotary embeddings need an even head width", ErrInvalidConfig, document.HeadWidth)
	case document.Activation != supportedActivation || document.GatedActivation:
		return fmt.Errorf("%w: activation %q (gated %t) is unsupported", ErrInvalidConfig, document.Activation, document.GatedActivation)
	case chronos.InputPatchSize != PatchSize || chronos.InputPatchStride != PatchSize || chronos.OutputPatchSize != PatchSize:
		return fmt.Errorf("%w: patch size %d, stride %d, and output patch size %d must all be %d",
			ErrInvalidConfig, chronos.InputPatchSize, chronos.InputPatchStride, chronos.OutputPatchSize, PatchSize)
	case document.LayerNormEpsilon < 0 || document.RopeTheta <= 0:
		return fmt.Errorf("%w: layer_norm_epsilon %g or rope_theta %g out of range", ErrInvalidConfig, document.LayerNormEpsilon, document.RopeTheta)
	}
	return validateQuantiles(chronos.Quantiles)
}

// validateQuantiles checks that levels is non-empty and strictly ascending
// within (0, 1).
func validateQuantiles(levels []float64) error {
	if len(levels) == 0 {
		return fmt.Errorf("%w: no quantile levels", ErrInvalidConfig)
	}
	for i, level := range levels {
		if level <= 0 || level >= 1 || (i > 0 && level <= levels[i-1]) {
			return fmt.Errorf("%w: quantile levels %v are not strictly ascending in (0, 1)", ErrInvalidConfig, levels)
		}
	}
	return nil
}
