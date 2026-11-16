package chronos2

import (
	"context"
	"fmt"
)

// Forecast predicts the next horizon steps of one history. The history holds
// observations at a fixed interval, oldest first, and the forecast continues
// at the same interval: horizon counts future steps, not units of time. NaN
// marks a missing observation, but at least one value must be observed. A
// history longer than [Model.ContextLength] (or the [WithContextLength]
// value) is trimmed to its most recent steps. The horizon must be at least 1
// and may not exceed [Model.MaxHorizon] unless [WithLongHorizon] is passed.
// The history is never modified or retained.
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first.
func (m *Model) Forecast(ctx context.Context, history []float64, horizon int, opts ...ForecastOption) (Forecast, error) {
	results, err := m.forecastTasks(ctx, []Task{{Targets: [][]float64{history}}}, horizon, opts,
		func(int, int) string { return "history" })
	if err != nil {
		return Forecast{}, err
	}
	return results[0][0], nil
}

// ForecastBatch predicts the next horizon steps of many independent
// histories, which may differ in length, returning one [Forecast] per history
// in input order. It is faster than calling [Model.Forecast] for each history,
// because the model processes up to [WithBatchSize] series per pass. Each
// history follows the rules of Forecast and forms its own task; pass
// [WithCrossLearning] to let the histories share information. An empty or
// nil histories slice returns an empty result.
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first.
func (m *Model) ForecastBatch(ctx context.Context, histories [][]float64, horizon int, opts ...ForecastOption) ([]Forecast, error) {
	tasks := make([]Task, len(histories))
	for i, history := range histories {
		tasks[i] = Task{Targets: [][]float64{history}}
	}
	results, err := m.forecastTasks(ctx, tasks, horizon, opts,
		func(task, _ int) string { return fmt.Sprintf("history %d", task) })
	if err != nil {
		return nil, err
	}
	forecasts := make([]Forecast, len(results))
	for i, result := range results {
		forecasts[i] = result[0]
	}
	return forecasts, nil
}

// ForecastMultivariate jointly predicts the next horizon steps of variates:
// several series measured on one time axis that move together, such as the
// temperature, humidity, and pressure at one weather station. Each variate's
// forecast draws on the others. The variates must share a length, and each
// follows the rules of [Model.Forecast]. It returns one [Forecast] per
// variate in input order.
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first; no variates returns an error wrapping ErrEmptyHistory.
func (m *Model) ForecastMultivariate(ctx context.Context, variates [][]float64, horizon int, opts ...ForecastOption) ([]Forecast, error) {
	results, err := m.forecastTasks(ctx, []Task{{Targets: variates}}, horizon, opts,
		func(_, target int) string { return fmt.Sprintf("variate %d", target) })
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// ForecastWithCovariates predicts the next horizon steps of one target
// informed by covariates: series that help predict the target without being
// forecast themselves. Past covariate series share the target's length;
// future covariate series hold exactly horizon values. See [Covariates] for
// the rules and [Model.Forecast] for the target's.
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first.
func (m *Model) ForecastWithCovariates(ctx context.Context, task CovariateTask, horizon int, opts ...ForecastOption) (Forecast, error) {
	general := Task{Targets: [][]float64{task.Target}, Covariates: task.Covariates}
	results, err := m.forecastTasks(ctx, []Task{general}, horizon, opts,
		func(int, int) string { return "target" })
	if err != nil {
		return Forecast{}, err
	}
	return results[0][0], nil
}

// ForecastTasks is the general forecasting method: any number of tasks, each
// with any number of targets and covariates, all forecast horizon steps
// ahead. It returns one slice per task, in input order, holding one
// [Forecast] per target in input order. Tasks are independent unless
// [WithCrossLearning] is passed, and may differ in length. An empty or nil
// tasks slice returns an empty result.
//
// Errors wrap one of the package sentinels, or the context's error when ctx
// ends first.
func (m *Model) ForecastTasks(ctx context.Context, tasks []Task, horizon int, opts ...ForecastOption) ([][]Forecast, error) {
	return m.forecastTasks(ctx, tasks, horizon, opts, labelTaskTarget)
}
