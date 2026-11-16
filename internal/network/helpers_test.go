package network

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// syntheticScenarios lists the synthetic checkpoint's stage fixtures.
var syntheticScenarios = []string{
	"single",
	"groups_gaps_covariates",
	"long",
	"constant_and_single_observation",
}

// syntheticConfig returns the synthetic checkpoint's parsed config, failing
// the test when it does not parse.
func syntheticConfig(tb testing.TB) Config {
	tb.Helper()
	configJSON, _ := testmodel.SyntheticCheckpoint(tb)
	config, err := ParseConfig(configJSON)
	if err != nil {
		tb.Fatalf("parse synthetic config: %v", err)
	}
	return config
}

// syntheticLookup returns a lookup over the synthetic checkpoint's tensors.
func syntheticLookup(tb testing.TB) TensorLookup {
	tb.Helper()
	_, shards := testmodel.SyntheticCheckpoint(tb)
	file, err := safetensors.Parse(shards[0])
	if err != nil {
		tb.Fatalf("parse synthetic weights: %v", err)
	}
	tensors := make(map[string]testmodel.Tensor, len(file.Tensors))
	for name, tensor := range file.Tensors {
		values, err := tensor.Float32s()
		if err != nil {
			tb.Fatalf("decode %s: %v", name, err)
		}
		tensors[name] = testmodel.Tensor{Shape: tensor.Shape, Values: values}
	}
	return func(name string) ([]float32, []int, bool) {
		tensor, ok := tensors[name]
		return tensor.Values, tensor.Shape, ok
	}
}

// syntheticNetwork builds the synthetic checkpoint's network, failing the
// test when construction fails.
func syntheticNetwork(tb testing.TB) *Network {
	tb.Helper()
	network, err := New(syntheticConfig(tb), syntheticLookup(tb))
	if err != nil {
		tb.Fatalf("build synthetic network: %v", err)
	}
	return network
}

// syntheticFixture reads the synthetic stage fixture for scenario.
func syntheticFixture(tb testing.TB, scenario string) testmodel.ForwardFixture {
	tb.Helper()
	return testmodel.ReadForwardFixture(tb, filepath.Join(testmodel.SyntheticDir(), "forward_"+scenario+".safetensors"))
}

// batchFromFixture converts a stage fixture's inputs into an engine batch.
func batchFromFixture(fixture testmodel.ForwardFixture) Batch {
	return Batch{
		Rows:             fixture.Rows,
		ContextLength:    fixture.ContextLength,
		Context:          fixture.Context,
		GroupIDs:         fixture.GroupIDs,
		OutputPatches:    fixture.OutputPatches,
		FutureCovariates: fixture.FutureCovariates,
	}
}

// stageSpy is a StageObserver that keeps a copy of every stage it sees.
type stageSpy map[string][]float32

// observe records a copy of values under name.
func (s stageSpy) observe(name string, values []float32) {
	s[name] = slices.Clone(values)
}

// singleRowBatch returns a valid one-row batch of length steps with one
// output patch, for tests that need any well-formed input.
func singleRowBatch(length int) Batch {
	context := make([]float32, length)
	for i := range context {
		context[i] = float32(i % 7)
	}
	future := make([]float32, 16)
	for i := range future {
		future[i] = nan32()
	}
	return Batch{Rows: 1, ContextLength: length, Context: context, GroupIDs: []int{0}, OutputPatches: 1, FutureCovariates: future}
}
