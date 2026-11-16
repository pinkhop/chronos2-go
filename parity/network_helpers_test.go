package parity

import (
	"path/filepath"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/network"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// forwardScenarios lists the real-model stage fixtures shared by both models.
var forwardScenarios = []string{
	"single",
	"groups_gaps_covariates",
	"long",
	"constant_and_single_observation",
}

// realNetwork builds the named real model ("small" or "base") from its
// committed weight shards, failing the test when it cannot.
func realNetwork(tb testing.TB, model string) *network.Network {
	tb.Helper()
	configJSON, lookup := testmodel.CheckpointTensors(tb, filepath.Join(modelModuleDir(tb, model), "weights"))
	config, err := network.ParseConfig(configJSON)
	if err != nil {
		tb.Fatalf("parse %s config: %v", model, err)
	}
	built, err := network.New(config, lookup)
	if err != nil {
		tb.Fatalf("build %s network: %v", model, err)
	}
	return built
}

// checkForwardStages runs one real-model stage fixture through net and
// compares every captured stage and the final quantiles with the reference.
func checkForwardStages(t *testing.T, net *network.Network, model, scenario string) {
	t.Helper()

	// Given
	fixture := testmodel.ReadForwardFixture(t, filepath.Join("testdata", model, "forward_"+scenario+".safetensors"))
	batch := network.Batch{
		Rows: fixture.Rows, ContextLength: fixture.ContextLength, Context: fixture.Context,
		GroupIDs: fixture.GroupIDs, OutputPatches: fixture.OutputPatches, FutureCovariates: fixture.FutureCovariates,
	}
	stages := map[string][]float32{}
	observe := func(name string, values []float32) { stages[name] = append([]float32(nil), values...) }

	// When
	output, err := net.ForwardObserved(t.Context(), batch, network.ForwardOptions{Parallelism: 4}, observe)
	// Then
	if err != nil {
		t.Fatalf("ForwardObserved: %v", err)
	}
	for _, name := range stageNames(net.Config().Layers) {
		testmodel.CompareStage(t, name, stages[name], fixture.Stages[name], testmodel.StageAbsTolerance, testmodel.StageRelTolerance)
	}
	want := fixture.Stages["quantiles"]
	perRow := len(want.Values) / fixture.Rows
	for i, got := range output.Values {
		abs := testmodel.ForecastAbsScale * float64(fixture.Stages["scale"].Values[i/perRow])
		if !testmodel.Within(float64(got), float64(want.Values[i]), abs, testmodel.ForecastRelTolerance) {
			t.Errorf("quantiles[%d] = %g, want %g", i, got, want.Values[i])
			return
		}
	}
}

// stageNames lists the captured intermediate stages in pipeline order.
func stageNames(layers int) []string {
	names := []string{"loc", "scale", "normalized_context", "patched_context", "patched_future", "input_embeds"}
	for layer := range layers {
		names = append(names, "layer_"+twoDigits(layer))
	}
	return append(names, "hidden", "quantiles_normalized")
}

// twoDigits formats a layer index the way the fixtures name layers.
func twoDigits(value int) string {
	return string(rune('0'+value/10)) + string(rune('0'+value%10))
}
