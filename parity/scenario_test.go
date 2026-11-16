package parity

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// Tolerances from DESIGN.md "Parity Fixtures".
const (
	// stageAbsTolerance is the absolute tolerance for intermediate tensors,
	// which include embeddings.
	stageAbsTolerance = 1e-4
	// stageRelTolerance is the relative tolerance for intermediate tensors.
	stageRelTolerance = 1e-4
	// forecastAbsToleranceScale multiplies a row's scale to give the
	// absolute tolerance for final quantiles.
	forecastAbsToleranceScale = 1e-4
	// forecastRelTolerance is the relative tolerance for final quantiles.
	forecastRelTolerance = 1e-3
)

// minimumScale is the scale the reference substitutes for a zero standard
// deviation.
const minimumScale = 1e-5

// knownDeviation identifies one task that DESIGN.md "Known Deviations"
// exempts from comparison with the reference.
type knownDeviation struct {
	// model names the checkpoint's fixture directory, "small" or "base".
	model string

	// scenario is the scenario's name.
	scenario string

	// task is the task's index within the scenario.
	task int
}

// knownDeviations lists every exempt task; it must match the table in
// DESIGN.md "Known Deviations".
var knownDeviations = map[knownDeviation]bool{
	{model: "small", scenario: "categorical_target_encoding", task: 1}: true,
	{model: "base", scenario: "categorical_target_encoding", task: 1}:  true,
}

// scenarioPaths returns every scenario manifest for one model's fixtures.
func scenarioPaths(t *testing.T, model string) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", model, "scenarios", "*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("list %s scenarios: %v (found %d)", model, err, len(paths))
	}
	return paths
}

// checkScenarios runs each of the model's scenarios whose tier matches slow
// and compares the results with the reference outputs.
func checkScenarios(t *testing.T, model string, newModel func() (*chronos2.Model, error), slow bool) {
	t.Helper()
	m, err := newModel()
	if err != nil {
		t.Fatalf("load %s model: %v", model, err)
	}
	for _, path := range scenarioPaths(t, model) {
		scenario, want := testmodel.ReadScenario(t, path)
		if scenario.Slow != slow {
			continue
		}
		t.Run(scenario.Name, func(t *testing.T) {
			t.Parallel()
			checkScenario(t, model, m, scenario, want)
		})
	}
}

// checkScenario runs one scenario through the public API and checks every
// task's forecasts against the reference. model names the checkpoint's
// fixture directory, which selects its known deviations.
func checkScenario(t *testing.T, model string, m *chronos2.Model, scenario testmodel.Scenario, want map[string]testmodel.Tensor) {
	t.Helper()
	if scenario.Method == "Embed" {
		checkEmbedding(t, m, scenario, want)
		return
	}

	// When: the scenario's method is called.
	got, err := runForecast(t.Context(), m, scenario)
	// Then: every task matches the reference, or is a plausible known
	// deviation.
	if err != nil {
		t.Fatalf("%s: %v", scenario.Method, err)
	}
	window := contextWindow(m, scenario)
	for taskIndex, task := range scenario.Tasks {
		exempt := knownDeviations[knownDeviation{model: model, scenario: scenario.Name, task: taskIndex}]
		checkTask(t, exempt, taskIndex, got[taskIndex], want[taskKey(taskIndex)], task.Targets, window)
	}
}

// runForecast calls the public method the scenario names and returns one
// slice of forecasts per task.
func runForecast(ctx context.Context, m *chronos2.Model, scenario testmodel.Scenario) ([][]chronos2.Forecast, error) {
	opts := forecastOptions(scenario.Options)
	tasks := scenario.Tasks
	switch scenario.Method {
	case "Forecast":
		fc, err := m.Forecast(ctx, tasks[0].Targets[0], scenario.Horizon, opts...)
		return [][]chronos2.Forecast{{fc}}, err
	case "ForecastBatch":
		histories := make([][]float64, len(tasks))
		for i, task := range tasks {
			histories[i] = task.Targets[0]
		}
		fcs, err := m.ForecastBatch(ctx, histories, scenario.Horizon, opts...)
		out := make([][]chronos2.Forecast, len(fcs))
		for i, fc := range fcs {
			out[i] = []chronos2.Forecast{fc}
		}
		return out, err
	case "ForecastMultivariate":
		fcs, err := m.ForecastMultivariate(ctx, seriesSlices(tasks[0].Targets), scenario.Horizon, opts...)
		return [][]chronos2.Forecast{fcs}, err
	case "ForecastWithCovariates":
		task := chronos2.CovariateTask{Target: tasks[0].Targets[0], Covariates: covariates(tasks[0])}
		fc, err := m.ForecastWithCovariates(ctx, task, scenario.Horizon, opts...)
		return [][]chronos2.Forecast{{fc}}, err
	default:
		apiTasks := make([]chronos2.Task, len(tasks))
		for i, task := range tasks {
			apiTasks[i] = chronos2.Task{Targets: seriesSlices(task.Targets), Covariates: covariates(task)}
		}
		return m.ForecastTasks(ctx, apiTasks, scenario.Horizon, opts...)
	}
}

// forecastOptions converts scenario options into forecast options.
func forecastOptions(options testmodel.ScenarioOptions) []chronos2.ForecastOption {
	var opts []chronos2.ForecastOption
	if options.ContextLength > 0 {
		opts = append(opts, chronos2.WithContextLength(options.ContextLength))
	}
	if options.BatchSize > 0 {
		opts = append(opts, chronos2.WithBatchSize(options.BatchSize))
	}
	if options.CrossLearning {
		opts = append(opts, chronos2.WithCrossLearning())
	}
	if options.LongHorizon {
		opts = append(opts, chronos2.WithLongHorizon())
	}
	return opts
}

// seriesSlices converts fixture series into plain slices.
func seriesSlices(series []testmodel.Series) [][]float64 {
	out := make([][]float64, len(series))
	for i, s := range series {
		out[i] = s
	}
	return out
}

// covariates converts a fixture task's covariates into the public type.
func covariates(task testmodel.ScenarioTask) chronos2.Covariates {
	plain := func(in map[string]testmodel.Series) map[string][]float64 {
		if len(in) == 0 {
			return nil
		}
		out := make(map[string][]float64, len(in))
		for name, s := range in {
			out[name] = s
		}
		return out
	}
	return chronos2.Covariates{
		Past:              plain(task.Past),
		Future:            plain(task.Future),
		PastCategorical:   task.PastCategorical,
		FutureCategorical: task.FutureCategorical,
	}
}

// checkTask checks one task's forecasts, one per target history, against
// the reference tensor [targets, quantiles, horizon]. A task that is not
// exempt must match within the forecast tolerance. An exempt task, a known
// deviation, must instead be finite with non-decreasing levels, and must
// still differ from the reference, so a stale exemption fails.
func checkTask(t *testing.T, exempt bool, taskIndex int, forecasts []chronos2.Forecast, want testmodel.Tensor, histories []testmodel.Series, window int) {
	t.Helper()
	if len(forecasts) != len(histories) || len(want.Shape) != 3 || want.Shape[0] != len(histories) {
		t.Fatalf("task %d: %d forecasts for %d targets, reference shape %v", taskIndex, len(forecasts), len(histories), want.Shape)
	}
	outside := 0
	for targetIndex, history := range histories {
		label := fmt.Sprintf("task %d target %d", taskIndex, targetIndex)
		scale := rowScale(history, window)
		lines := forecastLines(t, label, forecasts[targetIndex], want)
		deviation := testmodel.MeasureLines(lines, want, targetIndex, forecastAbsToleranceScale*scale, forecastRelTolerance)
		outside += deviation.Outside
		switch {
		case exempt:
			t.Logf("%s (known deviation): %d of %d values outside tolerance, max |error| / scale %.4g",
				label, deviation.Outside, deviation.Total, deviation.MaxError/scale)
			if err := errors.Join(testmodel.FiniteLines(lines), testmodel.OrderedLines(lines)); err != nil {
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
	percentiles := forecast.NativePercentiles()
	if len(percentiles) != levels || forecast.Horizon() != horizon {
		t.Fatalf("%s: %d levels × %d steps, want %d × %d", label, len(percentiles), forecast.Horizon(), levels, horizon)
	}
	lines := make([][]float64, levels)
	for level, percentile := range percentiles {
		line, err := forecast.Percentile(percentile)
		if err != nil {
			t.Fatalf("%s: Percentile(%v): %v", label, percentile, err)
		}
		lines[level] = line
	}
	return lines
}

// checkEmbedding runs Embed and compares its states, location, and scale with
// the reference.
func checkEmbedding(t *testing.T, m *chronos2.Model, scenario testmodel.Scenario, want map[string]testmodel.Tensor) {
	t.Helper()
	var opts []chronos2.EmbedOption
	if scenario.Options.ContextLength > 0 {
		opts = append(opts, chronos2.WithContextLength(scenario.Options.ContextLength))
	}

	// When: the history is embedded.
	embedding, err := m.Embed(t.Context(), scenario.Tasks[0].Targets[0], opts...)
	// Then: states, location, and scale match the reference.
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	expected := want["embedding"]
	states := embedding.States()
	if len(states) != expected.Shape[0] {
		t.Fatalf("got %d states, want %d", len(states), expected.Shape[0])
	}
	width := expected.Shape[1]
	mismatches := 0
	for i, state := range states {
		for j, value := range state {
			if !testmodel.Within(float64(value), float64(expected.Values[i*width+j]), stageAbsTolerance, stageRelTolerance) {
				mismatches++
			}
		}
	}
	if mismatches > 0 {
		t.Errorf("%d of %d embedding values outside tolerance", mismatches, len(states)*width)
	}
	if loc := float64(want["loc"].Values[0]); !testmodel.Within(embedding.Location(), loc, stageAbsTolerance, stageRelTolerance) {
		t.Errorf("Location() = %v, want %v", embedding.Location(), loc)
	}
	if scale := float64(want["scale"].Values[0]); !testmodel.Within(embedding.Scale(), scale, stageAbsTolerance, stageRelTolerance) {
		t.Errorf("Scale() = %v, want %v", embedding.Scale(), scale)
	}
}

// taskKey names the reference output tensor of task i.
func taskKey(i int) string {
	return fmt.Sprintf("task_%03d", i)
}

// contextWindow returns the context length a scenario's forecasts use.
func contextWindow(m *chronos2.Model, scenario testmodel.Scenario) int {
	if scenario.Options.ContextLength > 0 {
		return scenario.Options.ContextLength
	}
	return m.ContextLength()
}

// rowScale returns the normalization scale the model applies to a history
// trimmed to window steps: the population standard deviation of its observed
// values, 1 when none are observed, and minimumScale when it is zero.
func rowScale(history []float64, window int) float64 {
	if len(history) > window {
		history = history[len(history)-window:]
	}
	var sum, count float64
	for _, v := range history {
		if !math.IsNaN(v) {
			sum += v
			count++
		}
	}
	if count == 0 {
		return 1
	}
	mean := sum / count
	var squares float64
	for _, v := range history {
		if !math.IsNaN(v) {
			squares += (v - mean) * (v - mean)
		}
	}
	scale := math.Sqrt(squares / count)
	if scale == 0 {
		return minimumScale
	}
	return scale
}
