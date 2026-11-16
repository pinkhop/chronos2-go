package chronos2_test

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// Tolerances from DESIGN.md "Parity Fixtures". Final quantiles allow an
// absolute error of forecastAbsPerScale times the target's scale, computed
// from its trimmed history as the reference normalizes it; embeddings,
// locations, and scales are intermediate values and use the stage tolerance.
const (
	forecastAbsPerScale  = 1e-4
	forecastRelTolerance = 1e-3
	stageAbsTolerance    = 1e-4
	stageRelTolerance    = 1e-4

	// minimumScale replaces a zero standard deviation, as the reference does.
	minimumScale = 1e-5
)

// knownDeviation identifies one synthetic-checkpoint task that DESIGN.md
// "Known Deviations" exempts from comparison with the reference.
type knownDeviation struct {
	// scenario is the scenario's name.
	scenario string

	// task is the task's index within the scenario.
	task int
}

// knownDeviations lists every exempt synthetic-checkpoint task; it must
// match the table in DESIGN.md "Known Deviations".
var knownDeviations = map[knownDeviation]bool{
	{scenario: "categorical_target_encoding", task: 1}: true,
}

// TestModel_SyntheticScenarios_MatchReference verifies that a Model over the synthetic checkpoint matches the reference on every scenario.
func TestModel_SyntheticScenarios_MatchReference(t *testing.T) {
	t.Parallel()

	paths, err := filepath.Glob(filepath.Join(testmodel.SyntheticDir(), "scenarios", "*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no scenarios found: %v", err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			t.Parallel()

			// Given
			scenario, outputs := testmodel.ReadScenario(t, path)
			if scenario.Slow && !slowTier {
				t.Skip("slow scenario; run with -tags=slow")
			}
			model := loadSynthetic(t)

			// When
			result, err := runScenario(t, model, scenario)
			// Then
			if err != nil {
				t.Fatalf("%s() error = %v", scenario.Method, err)
			}
			result.compare(t, scenario, outputs)
		})
	}
}

// loadSynthetic loads the synthetic checkpoint, failing the test on error.
func loadSynthetic(t *testing.T) *chronos2.Model {
	t.Helper()
	config, shards := testmodel.SyntheticCheckpoint(t)
	model, err := chronos2.Load(chronos2.Checkpoint{Name: "synthetic", Config: config, Shards: shards})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	return model
}

// scenarioResult holds what a scenario's call returned: forecasts per task
// and target, or an embedding.
type scenarioResult struct {
	// forecasts holds one slice per task with one Forecast per target.
	forecasts [][]chronos2.Forecast

	// embedding is set for Embed scenarios.
	embedding chronos2.Embedding
}

// compare checks the result against the scenario's expected outputs.
func (r scenarioResult) compare(t *testing.T, scenario testmodel.Scenario, outputs map[string]testmodel.Tensor) {
	t.Helper()
	if scenario.Method == "Embed" {
		compareEmbedding(t, r.embedding, outputs)
		return
	}
	window := scenario.Options.ContextLength
	if window == 0 {
		window = math.MaxInt
	}
	for taskIndex, forecasts := range r.forecasts {
		exempt := knownDeviations[knownDeviation{scenario: scenario.Name, task: taskIndex}]
		checkTask(t, exempt, taskIndex, forecasts, outputs[fmt.Sprintf("task_%03d", taskIndex)], scenario.Tasks[taskIndex].Targets, window)
	}
}

// checkTask checks one task's forecasts, one per target history, against
// the reference tensor [targets, quantiles, horizon]. A task that is not
// exempt must match within the forecast tolerance. An exempt task, a known
// deviation, must instead be finite and must still differ from the
// reference, so a stale exemption fails. The synthetic checkpoint's random
// weights do not order quantile levels, even in the reference, so unlike the
// parity module this harness does not check their order.
func checkTask(t *testing.T, exempt bool, taskIndex int, forecasts []chronos2.Forecast, want testmodel.Tensor, histories []testmodel.Series, window int) {
	t.Helper()
	if len(forecasts) != len(histories) || len(want.Shape) != 3 || want.Shape[0] != len(histories) {
		t.Fatalf("task %d: %d forecasts for %d targets, reference shape %v", taskIndex, len(forecasts), len(histories), want.Shape)
	}
	outside := 0
	for targetIndex, history := range histories {
		label := fmt.Sprintf("task %d target %d", taskIndex, targetIndex)
		scale := historyScale(history, window)
		lines := forecastLines(t, label, forecasts[targetIndex], want)
		deviation := testmodel.MeasureLines(lines, want, targetIndex, forecastAbsPerScale*scale, forecastRelTolerance)
		outside += deviation.Outside
		switch {
		case exempt:
			t.Logf("%s (known deviation): %d of %d values outside tolerance, max |error| / scale %.4g",
				label, deviation.Outside, deviation.Total, deviation.MaxError/scale)
			if err := testmodel.FiniteLines(lines); err != nil {
				t.Errorf("%s (known deviation): %v", label, err)
			}
		case deviation.Outside > 0:
			t.Errorf("%s: %d of %d values outside tolerance (max |error| / scale %.4g); first: %s",
				label, deviation.Outside, deviation.Total, deviation.MaxError/scale, deviation.First)
		}
	}
	if exempt && outside == 0 {
		t.Errorf("task %d is a known deviation but matches the reference within tolerance; remove the exemption", taskIndex)
	}
}

// forecastLines returns forecast's native percentile lines, one per level,
// failing the test unless their count and length match want, a [targets,
// quantiles, horizon] reference tensor.
func forecastLines(t *testing.T, label string, forecast chronos2.Forecast, want testmodel.Tensor) [][]float64 {
	t.Helper()
	levels, horizon := want.Shape[1], want.Shape[2]
	percents := forecast.NativePercentiles()
	if len(percents) != levels || forecast.Horizon() != horizon {
		t.Fatalf("%s: %d levels × %d steps, want %d × %d", label, len(percents), forecast.Horizon(), levels, horizon)
	}
	lines := make([][]float64, levels)
	for level, percent := range percents {
		line, err := forecast.Percentile(percent)
		if err != nil {
			t.Fatalf("%s: Percentile(%v) error = %v", label, percent, err)
		}
		lines[level] = line
	}
	return lines
}

// compareEmbedding checks an embedding's states, location, and scale.
func compareEmbedding(t *testing.T, embedding chronos2.Embedding, outputs map[string]testmodel.Tensor) {
	t.Helper()
	want := outputs["embedding"]
	states := embedding.States()
	if len(states) != want.Shape[0] || len(states[0]) != want.Shape[1] {
		t.Fatalf("states %d×%d, want %v", len(states), len(states[0]), want.Shape)
	}
	mismatches := 0
	for token, state := range states {
		for element, got := range state {
			expected := float64(want.Values[token*want.Shape[1]+element])
			if !testmodel.Within(float64(got), expected, stageAbsTolerance, stageRelTolerance) && mismatches < 5 {
				mismatches++
				t.Errorf("state %d element %d = %v, want %v", token, element, got, expected)
			}
		}
	}
	if location := float64(outputs["loc"].Values[0]); !testmodel.Within(embedding.Location(), location, stageAbsTolerance, stageRelTolerance) {
		t.Errorf("Location() = %v, want %v", embedding.Location(), location)
	}
	if scale := float64(outputs["scale"].Values[0]); !testmodel.Within(embedding.Scale(), scale, stageAbsTolerance, stageRelTolerance) {
		t.Errorf("Scale() = %v, want %v", embedding.Scale(), scale)
	}
}

// historyScale returns the population standard deviation of the observed
// values among the last window steps of history, the scale the reference
// normalizes a target row by: 1 with no observations, minimumScale when 0.
func historyScale(history []float64, window int) float64 {
	trimmed := history[len(history)-min(window, len(history)):]
	var sum, count float64
	for _, value := range trimmed {
		if !math.IsNaN(value) {
			sum, count = sum+value, count+1
		}
	}
	if count == 0 {
		return 1
	}
	mean := sum / count
	var squares float64
	for _, value := range trimmed {
		if !math.IsNaN(value) {
			squares += (value - mean) * (value - mean)
		}
	}
	if scale := math.Sqrt(squares / count); scale > 0 {
		return scale
	}
	return minimumScale
}
