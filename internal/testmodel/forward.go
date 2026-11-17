package testmodel

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// Stage tolerances for intermediate tensors, as DESIGN.md "Parity Fixtures"
// defines them.
const (
	// StageAbsTolerance is the absolute tolerance for intermediate tensors.
	StageAbsTolerance = 1e-4
	// StageRelTolerance is the relative tolerance for intermediate tensors.
	StageRelTolerance = 1e-4
	// ForecastRelTolerance is the relative tolerance for final quantiles;
	// their absolute tolerance is ForecastAbsScale times the row's scale.
	ForecastRelTolerance = 1e-3
	// ForecastAbsScale multiplies a row's scale to give the absolute
	// tolerance for its final quantiles.
	ForecastAbsScale = 1e-4
)

// ForwardFixture is one intermediate-stage fixture: the inputs of a single
// forward pass and every stage the reference captured, read from a
// forward_<scenario>.safetensors file and its JSON manifest.
type ForwardFixture struct {
	// Rows is the number of rows in the batch, R.
	Rows int
	// ContextLength is the number of steps per row, L.
	ContextLength int
	// OutputPatches is the number of predicted patches, P.
	OutputPatches int
	// Context holds Rows × ContextLength values with NaN for missing steps.
	Context []float32
	// GroupIDs holds one group per row.
	GroupIDs []int
	// FutureCovariates holds Rows × 16·OutputPatches values with NaN for
	// unknown steps.
	FutureCovariates []float32
	// Stages maps each captured stage name (loc, scale, normalized_context,
	// patched_context, patched_future, input_embeds, layer_00, …, hidden,
	// quantiles_normalized, quantiles) to its tensor.
	Stages map[string]Tensor
}

// forwardManifest is the part of a forward fixture's JSON manifest the tests
// read.
type forwardManifest struct {
	// OutputPatches is the number of predicted patches.
	OutputPatches int `json:"output_patches"`
}

// ReadForwardFixture reads the stage fixture at path (a .safetensors file
// whose manifest sits beside it with a .json extension). It fails the test
// when either file is missing or malformed.
func ReadForwardFixture(tb testing.TB, path string) ForwardFixture {
	tb.Helper()
	// #nosec G304 -- tests pass fixed repository fixture paths.
	manifestData, err := os.ReadFile(strings.TrimSuffix(path, ".safetensors") + ".json")
	if err != nil {
		tb.Fatalf("read forward manifest: %v", err)
	}
	var manifest forwardManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		tb.Fatalf("decode forward manifest: %v", err)
	}
	tensors := ReadTensors(tb, path)
	context, ok := tensors["context"]
	if !ok || len(context.Shape) != 2 {
		tb.Fatalf("forward fixture %s lacks a 2-D context tensor", path)
	}
	groups := tensors["group_ids"].Values
	groupIDs := make([]int, len(groups))
	for i, group := range groups {
		groupIDs[i] = int(group)
	}
	return ForwardFixture{
		Rows:             context.Shape[0],
		ContextLength:    context.Shape[1],
		OutputPatches:    manifest.OutputPatches,
		Context:          context.Values,
		GroupIDs:         groupIDs,
		FutureCovariates: tensors["future_covariates"].Values,
		Stages:           tensors,
	}
}

// CompareStage fails the test when got differs from the fixture stage want
// in length or in any element beyond |got − want| ≤ abs + rel·|want|. It
// reports the first mismatch, the count of mismatches, and the largest
// absolute error so a failure localizes quickly.
func CompareStage(tb testing.TB, name string, got []float32, want Tensor, abs, rel float64) {
	tb.Helper()
	if len(got) != len(want.Values) {
		tb.Errorf("stage %s: got %d values, want %d (shape %v)", name, len(got), len(want.Values), want.Shape)
		return
	}
	mismatches, first, worst := 0, -1, 0.0
	for i := range got {
		if Within(float64(got[i]), float64(want.Values[i]), abs, rel) {
			continue
		}
		mismatches++
		if first < 0 {
			first = i
		}
		worst = max(worst, math.Abs(float64(got[i])-float64(want.Values[i])))
	}
	if mismatches > 0 {
		tb.Errorf("stage %s: %d of %d values outside tolerance; first at %v: got %g, want %g; worst abs error %g",
			name, mismatches, len(got), unflatten(first, want.Shape), got[first], want.Values[first], worst)
	}
}

// unflatten converts a flat row-major index into per-dimension indices for
// failure messages.
func unflatten(index int, shape []int) []int {
	indices := make([]int, len(shape))
	for d := len(shape) - 1; d >= 0; d-- {
		if shape[d] > 0 {
			indices[d] = index % shape[d]
			index /= shape[d]
		}
	}
	return indices
}

// CheckpointTensors reads config.json from configDir and every safetensors
// shard in shardDirs, and returns the config together with a lookup over
// every tensor, decoded to float32 copies. configDir is the weights directory
// of a real model module, or any directory holding a config.json; shardDirs
// are the directories holding the checkpoint's shards, which may include
// configDir, and must hold at least one shard in all. The lookup has the
// signature of network.TensorLookup. It fails the test when a file cannot be
// read or parsed.
func CheckpointTensors(tb testing.TB, configDir string, shardDirs ...string) (config []byte, lookup func(name string) ([]float32, []int, bool)) {
	tb.Helper()
	// #nosec G304 -- tests pass fixed repository paths; no untrusted input.
	config, err := os.ReadFile(filepath.Join(configDir, "config.json"))
	if err != nil {
		tb.Fatalf("read checkpoint config: %v", err)
	}
	var shardPaths []string
	for _, dir := range shardDirs {
		paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
		if err != nil {
			tb.Fatalf("find checkpoint shards in %s: %v", dir, err)
		}
		slices.Sort(paths)
		shardPaths = append(shardPaths, paths...)
	}
	if len(shardPaths) == 0 {
		tb.Fatalf("find checkpoint shards: none in %v", shardDirs)
	}
	tensors := make(map[string]Tensor)
	for _, path := range shardPaths {
		if err := readShard(path, tensors); err != nil {
			tb.Fatalf("read checkpoint shard: %v", err)
		}
	}
	return config, func(name string) ([]float32, []int, bool) {
		tensor, ok := tensors[name]
		return tensor.Values, tensor.Shape, ok
	}
}

// readShard decodes every tensor of the shard at path into tensors.
func readShard(path string, tensors map[string]Tensor) error {
	// #nosec G304 -- path comes from a glob over a fixed repository directory.
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	file, err := safetensors.Parse(data)
	if err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	for name, tensor := range file.Tensors {
		values, err := tensor.Float32s()
		if err != nil {
			return fmt.Errorf("decode %s in %s: %w", name, path, err)
		}
		tensors[name] = Tensor{Shape: tensor.Shape, Values: values}
	}
	return nil
}
