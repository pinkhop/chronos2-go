package chronos2_test

import (
	"fmt"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// runScenario calls the public method a scenario names with its inputs and
// options.
func runScenario(t *testing.T, model *chronos2.Model, scenario testmodel.Scenario) (scenarioResult, error) {
	t.Helper()
	ctx, horizon, options := t.Context(), scenario.Horizon, forecastOptions(scenario.Options)
	tasks := toTasks(scenario.Tasks)
	switch scenario.Method {
	case "Forecast":
		forecast, err := model.Forecast(ctx, tasks[0].Targets[0], horizon, options...)
		return scenarioResult{forecasts: [][]chronos2.Forecast{{forecast}}}, err
	case "ForecastBatch":
		histories := make([][]float64, len(tasks))
		for i, task := range tasks {
			histories[i] = task.Targets[0]
		}
		forecasts, err := model.ForecastBatch(ctx, histories, horizon, options...)
		result := scenarioResult{forecasts: make([][]chronos2.Forecast, len(forecasts))}
		for i, forecast := range forecasts {
			result.forecasts[i] = []chronos2.Forecast{forecast}
		}
		return result, err
	case "ForecastMultivariate":
		forecasts, err := model.ForecastMultivariate(ctx, tasks[0].Targets, horizon, options...)
		return scenarioResult{forecasts: [][]chronos2.Forecast{forecasts}}, err
	case "ForecastWithCovariates":
		task := chronos2.CovariateTask{Target: tasks[0].Targets[0], Covariates: tasks[0].Covariates}
		forecast, err := model.ForecastWithCovariates(ctx, task, horizon, options...)
		return scenarioResult{forecasts: [][]chronos2.Forecast{{forecast}}}, err
	case "ForecastTasks":
		forecasts, err := model.ForecastTasks(ctx, tasks, horizon, options...)
		return scenarioResult{forecasts: forecasts}, err
	case "Embed":
		embedding, err := model.Embed(ctx, tasks[0].Targets[0], embedOptions(scenario.Options)...)
		return scenarioResult{embedding: embedding}, err
	default:
		return scenarioResult{}, fmt.Errorf("unknown scenario method %q", scenario.Method)
	}
}

// forecastOptions converts a scenario's options to forecasting options.
func forecastOptions(options testmodel.ScenarioOptions) []chronos2.ForecastOption {
	var converted []chronos2.ForecastOption
	if options.ContextLength > 0 {
		converted = append(converted, chronos2.WithContextLength(options.ContextLength))
	}
	if options.BatchSize > 0 {
		converted = append(converted, chronos2.WithBatchSize(options.BatchSize))
	}
	if options.CrossLearning {
		converted = append(converted, chronos2.WithCrossLearning())
	}
	if options.LongHorizon {
		converted = append(converted, chronos2.WithLongHorizon())
	}
	return converted
}

// embedOptions converts a scenario's options to Embed options.
func embedOptions(options testmodel.ScenarioOptions) []chronos2.EmbedOption {
	if options.ContextLength > 0 {
		return []chronos2.EmbedOption{chronos2.WithContextLength(options.ContextLength)}
	}
	return nil
}

// toTasks converts scenario tasks to public tasks.
func toTasks(scenarioTasks []testmodel.ScenarioTask) []chronos2.Task {
	tasks := make([]chronos2.Task, len(scenarioTasks))
	for i, scenarioTask := range scenarioTasks {
		targets := make([][]float64, len(scenarioTask.Targets))
		for j, target := range scenarioTask.Targets {
			targets[j] = target
		}
		tasks[i] = chronos2.Task{Targets: targets, Covariates: chronos2.Covariates{
			Past:              toSeriesMap(scenarioTask.Past),
			Future:            toSeriesMap(scenarioTask.Future),
			PastCategorical:   scenarioTask.PastCategorical,
			FutureCategorical: scenarioTask.FutureCategorical,
		}}
	}
	return tasks
}

// toSeriesMap converts a scenario's numeric covariates, returning nil for
// none.
func toSeriesMap(series map[string]testmodel.Series) map[string][]float64 {
	if len(series) == 0 {
		return nil
	}
	converted := make(map[string][]float64, len(series))
	for name, values := range series {
		converted[name] = values
	}
	return converted
}
