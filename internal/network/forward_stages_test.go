package network

import (
	"testing"

	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// stageOrder lists the captured stages in pipeline order, so the first
// reported mismatch is the earliest divergence.
func stageOrder(layers int) []string {
	names := []string{"loc", "scale", "normalized_context", "patched_context", "patched_future", "input_embeds"}
	for layer := range layers {
		names = append(names, layerStageName(layer))
	}
	return append(names, "hidden", "quantiles_normalized")
}

// TestForwardObserved_SyntheticScenarios_MatchReferenceStages verifies that every observed forward stage matches the reference's intermediate values on the synthetic scenarios.
func TestForwardObserved_SyntheticScenarios_MatchReferenceStages(t *testing.T) {
	t.Parallel()
	for _, scenario := range syntheticScenarios {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()

			// Given
			network := syntheticNetwork(t)
			fixture := syntheticFixture(t, scenario)
			spy := stageSpy{}

			// When
			output, err := network.ForwardObserved(t.Context(), batchFromFixture(fixture), ForwardOptions{Parallelism: 3}, spy.observe)
			// Then
			if err != nil {
				t.Fatalf("ForwardObserved: %v", err)
			}
			for _, name := range stageOrder(network.config.Layers) {
				testmodel.CompareStage(t, name, spy[name], fixture.Stages[name], testmodel.StageAbsTolerance, testmodel.StageRelTolerance)
			}
			compareForecast(t, output, fixture)
		})
	}
}

// compareForecast checks output's denormalized quantiles against the
// fixture's with the forecast tolerance, whose absolute part scales with each
// row's scale.
func compareForecast(t *testing.T, output Output, fixture testmodel.ForwardFixture) {
	t.Helper()
	want := fixture.Stages["quantiles"]
	if len(output.Values) != len(want.Values) {
		t.Fatalf("quantiles: got %d values, want %d", len(output.Values), len(want.Values))
	}
	perRow := len(want.Values) / fixture.Rows
	scales := fixture.Stages["scale"].Values
	for i, got := range output.Values {
		abs := testmodel.ForecastAbsScale * float64(scales[i/perRow])
		if !testmodel.Within(float64(got), float64(want.Values[i]), abs, testmodel.ForecastRelTolerance) {
			t.Errorf("quantiles[%d] = %g, want %g", i, got, want.Values[i])
			return
		}
	}
}

// TestForward_TinyBlocks_MatchReferenceStages verifies that Forward, with tiny work blocks, still matches the reference stages.
func TestForward_TinyBlocks_MatchReferenceStages(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	fixture := syntheticFixture(t, "long")
	spy := stageSpy{}

	// When
	output, err := network.forward(t.Context(), batchFromFixture(fixture), ForwardOptions{Parallelism: 2}, spy.observe, passTuning{blockTokens: 5})
	// Then
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	for _, name := range stageOrder(network.config.Layers) {
		testmodel.CompareStage(t, name, spy[name], fixture.Stages[name], testmodel.StageAbsTolerance, testmodel.StageRelTolerance)
	}
	compareForecast(t, output, fixture)
}
