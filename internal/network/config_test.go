package network

import (
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// TestParseConfig_SyntheticCheckpoint_ReadsEveryField verifies that ParseConfig reads every field of the synthetic checkpoint's configuration.
func TestParseConfig_SyntheticCheckpoint_ReadsEveryField(t *testing.T) {
	t.Parallel()

	// Given
	configJSON, _ := testmodel.SyntheticCheckpoint(t)

	// When
	config, err := ParseConfig(configJSON)
	// Then
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	want := Config{
		ModelWidth: 32, FeedForwardWidth: 64, HeadWidth: 16, Heads: 2, Layers: 2,
		LayerNormEpsilon: 1e-6, RopeTheta: 10000, VocabularySize: 2, RegTokenID: 1,
		ContextLength: 8192, PatchSize: 16, MaxOutputPatches: 64,
		TimeEncodingScale: 8192, UseArcsinh: true, UseRegToken: true,
	}
	quantiles := config.Quantiles
	config.Quantiles = nil
	if !reflect.DeepEqual(config, want) {
		t.Errorf("config = %+v, want %+v", config, want)
	}
	if len(quantiles) != 13 || quantiles[0] != 0.01 || quantiles[6] != 0.5 || !slices.IsSorted(quantiles) {
		t.Errorf("Quantiles = %v, want 13 ascending levels from 0.01", quantiles)
	}
}

// TestParseConfig_InvalidDocuments_ReturnErrInvalidConfig verifies that ParseConfig returns ErrInvalidConfig for invalid documents.
// TestParseConfig_WithoutRegToken_UsesOneEntryVocabulary verifies that
// ParseConfig gives a checkpoint without a [REG] token a one-entry shared
// vocabulary, as the reference sets vocab_size to 1 when use_reg_token is
// false.
func TestParseConfig_WithoutRegToken_UsesOneEntryVocabulary(t *testing.T) {
	t.Parallel()

	// Given a config with use_reg_token set to false
	configJSON, _ := testmodel.SyntheticCheckpoint(t)
	document := strings.Replace(string(configJSON), `"use_reg_token": true`, `"use_reg_token": false`, 1)
	if document == string(configJSON) {
		t.Fatalf("synthetic config has no use_reg_token: true to replace")
	}

	// When it is parsed
	config, err := ParseConfig([]byte(document))
	// Then the vocabulary has one entry and no [REG] token
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if config.UseRegToken || config.VocabularySize != 1 || config.RegTokenID != 0 {
		t.Errorf("UseRegToken = %t, VocabularySize = %d, RegTokenID = %d; want false, 1, 0",
			config.UseRegToken, config.VocabularySize, config.RegTokenID)
	}
}

func TestParseConfig_InvalidDocuments_ReturnErrInvalidConfig(t *testing.T) {
	t.Parallel()
	configJSON, _ := testmodel.SyntheticCheckpoint(t)
	valid := string(configJSON)
	cases := map[string]string{
		"not JSON":            "{",
		"zero width":          strings.Replace(valid, `"d_model": 32`, `"d_model": 0`, 1),
		"odd head width":      strings.Replace(valid, `"d_kv": 16`, `"d_kv": 15`, 1),
		"gated activation":    strings.Replace(valid, `"is_gated_act": false`, `"is_gated_act": true`, 1),
		"other activation":    strings.Replace(valid, `"dense_act_fn": "relu"`, `"dense_act_fn": "gelu"`, 1),
		"unequal patch sizes": strings.Replace(valid, `"output_patch_size": 16`, `"output_patch_size": 8`, 1),
		"overlapping patches": strings.Replace(valid, `"input_patch_stride": 16`, `"input_patch_stride": 8`, 1),
		"patch size not 16": strings.NewReplacer(`"input_patch_size": 16`, `"input_patch_size": 8`,
			`"input_patch_stride": 16`, `"input_patch_stride": 8`, `"output_patch_size": 16`, `"output_patch_size": 8`).Replace(valid),
		"no quantiles":         strings.Replace(valid, `"quantiles": [`, `"quantiles": [], "unused": [`, 1),
		"unsorted quantiles":   strings.Replace(valid, `0.01,`, `0.6,`, 1),
		"zero context length":  strings.Replace(valid, `"context_length": 8192`, `"context_length": 0`, 1),
		"missing chronos part": strings.Replace(valid, `"chronos_config"`, `"other_config"`, 1),
	}
	for name, document := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// When
			_, err := ParseConfig([]byte(document))

			// Then
			if !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("err = %v, want ErrInvalidConfig", err)
			}
		})
	}
}

// TestNew_MissingTensor_ReturnsErrInvalidWeights verifies that New returns ErrInvalidWeights when a tensor is missing.
func TestNew_MissingTensor_ReturnsErrInvalidWeights(t *testing.T) {
	t.Parallel()

	// Given
	config := syntheticConfig(t)
	lookup := syntheticLookup(t)
	missing := func(name string) ([]float32, []int, bool) {
		if name == "encoder.block.1.layer.1.self_attention.v.weight" {
			return nil, nil, false
		}
		return lookup(name)
	}

	// When
	_, err := New(config, missing)

	// Then
	if !errors.Is(err, ErrInvalidWeights) {
		t.Errorf("err = %v, want ErrInvalidWeights", err)
	}
}

// TestNew_WrongShape_ReturnsErrInvalidWeights verifies that New returns ErrInvalidWeights for a tensor of the wrong shape.
func TestNew_WrongShape_ReturnsErrInvalidWeights(t *testing.T) {
	t.Parallel()

	// Given
	config := syntheticConfig(t)
	lookup := syntheticLookup(t)
	reshaped := func(name string) ([]float32, []int, bool) {
		data, shape, ok := lookup(name)
		if name == "output_patch_embedding.output_layer.weight" {
			return data, []int{shape[1], shape[0]}, ok
		}
		return data, shape, ok
	}

	// When
	_, err := New(config, reshaped)

	// Then
	if !errors.Is(err, ErrInvalidWeights) {
		t.Errorf("err = %v, want ErrInvalidWeights", err)
	}
}

// TestNew_MismatchedQuantileCount_ReturnsErrInvalidWeights verifies that New returns ErrInvalidWeights when the output layer's quantile count disagrees with the configuration.
func TestNew_MismatchedQuantileCount_ReturnsErrInvalidWeights(t *testing.T) {
	t.Parallel()

	// Given
	config := syntheticConfig(t)
	config.Quantiles = config.Quantiles[:5]

	// When
	_, err := New(config, syntheticLookup(t))

	// Then
	if !errors.Is(err, ErrInvalidWeights) {
		t.Errorf("err = %v, want ErrInvalidWeights", err)
	}
}

// TestNew_ImpossibleConfig_ReturnsErrInvalidConfig verifies that New returns ErrInvalidConfig for an impossible configuration, such as a negative layer count.
func TestNew_ImpossibleConfig_ReturnsErrInvalidConfig(t *testing.T) {
	t.Parallel()

	// Given
	config := syntheticConfig(t)
	config.Layers = -1

	// When
	_, err := New(config, syntheticLookup(t))

	// Then
	if !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("err = %v, want ErrInvalidConfig", err)
	}
}
