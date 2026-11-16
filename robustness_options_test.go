package chronos2_test

import (
	"errors"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// TestModel_Forecast_OptionAndHorizonEdges_ReturnExpectedError verifies that Model.Forecast accepts or rejects boundary option and horizon values as documented.
func TestModel_Forecast_OptionAndHorizonEdges_ReturnExpectedError(t *testing.T) {
	t.Parallel()

	model := synthetic(t)
	maxContext, maxHorizon := model.ContextLength(), model.MaxHorizon()
	tests := []struct {
		name    string
		horizon int
		options []chronos2.ForecastOption
		want    error
	}{
		{"context length 0", 8, []chronos2.ForecastOption{chronos2.WithContextLength(0)}, chronos2.ErrInvalidOption},
		{"context length 1", 8, []chronos2.ForecastOption{chronos2.WithContextLength(1)}, nil},
		{"context length at maximum", 8, []chronos2.ForecastOption{chronos2.WithContextLength(maxContext)}, nil},
		{"context length above maximum", 8, []chronos2.ForecastOption{chronos2.WithContextLength(maxContext + 1)}, chronos2.ErrInvalidOption},
		{"batch size 0", 8, []chronos2.ForecastOption{chronos2.WithBatchSize(0)}, chronos2.ErrInvalidOption},
		{"batch size 1", 8, []chronos2.ForecastOption{chronos2.WithBatchSize(1)}, nil},
		{"parallelism 0", 8, []chronos2.ForecastOption{chronos2.WithMaxParallelism(0)}, chronos2.ErrInvalidOption},
		{"parallelism negative", 8, []chronos2.ForecastOption{chronos2.WithMaxParallelism(-3)}, chronos2.ErrInvalidOption},
		{"horizon 0", 0, nil, chronos2.ErrInvalidHorizon},
		{"horizon negative", -1, nil, chronos2.ErrInvalidHorizon},
		{"horizon at maximum", maxHorizon, nil, nil},
		{"horizon above maximum", maxHorizon + 1, nil, chronos2.ErrHorizonTooLong},
		{"horizon above maximum with long horizon", maxHorizon + 1, []chronos2.ForecastOption{chronos2.WithLongHorizon()}, nil},
		{"nil option ignored", 8, []chronos2.ForecastOption{nil}, nil},
		{"zero shared option ignored", 8, []chronos2.ForecastOption{chronos2.SharedOption{}}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt

			// When
			forecast, err := model.Forecast(t.Context(), wave(100, 0), tt.horizon, tt.options...)

			// Then
			if tt.want == nil {
				if err != nil || forecast.Horizon() != tt.horizon {
					t.Errorf("Forecast() = horizon %d, error %v; want horizon %d, nil", forecast.Horizon(), err, tt.horizon)
				}
				return
			}
			if !errors.Is(err, tt.want) {
				t.Errorf("Forecast() error = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestModel_Embed_OptionEdges_ReturnExpectedError verifies that Model.Embed accepts or rejects boundary option values as documented.
func TestModel_Embed_OptionEdges_ReturnExpectedError(t *testing.T) {
	t.Parallel()

	model := synthetic(t)
	tests := []struct {
		name    string
		options []chronos2.EmbedOption
		want    error
	}{
		{"context length 0", []chronos2.EmbedOption{chronos2.WithContextLength(0)}, chronos2.ErrInvalidOption},
		{"context length above maximum", []chronos2.EmbedOption{chronos2.WithContextLength(model.ContextLength() + 1)}, chronos2.ErrInvalidOption},
		{"parallelism 0", []chronos2.EmbedOption{chronos2.WithMaxParallelism(0)}, chronos2.ErrInvalidOption},
		{"context length 1", []chronos2.EmbedOption{chronos2.WithContextLength(1)}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt

			// When
			_, err := model.Embed(t.Context(), wave(100, 0), tt.options...)

			// Then
			if (tt.want == nil && err != nil) || (tt.want != nil && !errors.Is(err, tt.want)) {
				t.Errorf("Embed() error = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestModel_ForecastTasks_CrossLearningAboveBatchSize_ReturnsBatchTooLarge verifies that a cross-learning call with more rows than the batch size returns ErrCrossLearningBatchTooLarge.
func TestModel_ForecastTasks_CrossLearningAboveBatchSize_ReturnsBatchTooLarge(t *testing.T) {
	t.Parallel()

	// Given three single-target tasks (three rows) and a batch size of two.
	model := synthetic(t)
	tasks := []chronos2.Task{{Targets: [][]float64{wave(50, 0)}}, {Targets: [][]float64{wave(60, 1)}}, {Targets: [][]float64{wave(70, 2)}}}

	// When
	_, err := model.ForecastTasks(t.Context(), tasks, 8, chronos2.WithCrossLearning(), chronos2.WithBatchSize(2))

	// Then
	if !errors.Is(err, chronos2.ErrCrossLearningBatchTooLarge) {
		t.Errorf("ForecastTasks() error = %v, want ErrCrossLearningBatchTooLarge", err)
	}
}

// TestModel_ForecastTasks_CrossLearningCovariateRowsAboveBatchSize_ReturnsBatchTooLarge verifies that covariate rows count toward the cross-learning batch-size limit.
func TestModel_ForecastTasks_CrossLearningCovariateRowsAboveBatchSize_ReturnsBatchTooLarge(t *testing.T) {
	t.Parallel()

	// Given two tasks whose covariates bring the call to four rows.
	model := synthetic(t)
	covariates := chronos2.Covariates{Past: map[string][]float64{"x": wave(50, 3)}}
	tasks := []chronos2.Task{{Targets: [][]float64{wave(50, 0)}, Covariates: covariates}, {Targets: [][]float64{wave(50, 1)}, Covariates: covariates}}

	// When
	_, err := model.ForecastTasks(t.Context(), tasks, 8, chronos2.WithCrossLearning(), chronos2.WithBatchSize(3))

	// Then
	if !errors.Is(err, chronos2.ErrCrossLearningBatchTooLarge) {
		t.Errorf("ForecastTasks() error = %v, want ErrCrossLearningBatchTooLarge", err)
	}
}

// TestModel_ForecastTasks_CrossLearningAtBatchSize_Succeeds verifies that a cross-learning call with exactly the batch size of rows succeeds.
func TestModel_ForecastTasks_CrossLearningAtBatchSize_Succeeds(t *testing.T) {
	t.Parallel()

	// Given exactly as many rows as the batch size.
	model := synthetic(t)
	tasks := []chronos2.Task{{Targets: [][]float64{wave(50, 0)}}, {Targets: [][]float64{wave(60, 1)}}}

	// When
	results, err := model.ForecastTasks(t.Context(), tasks, 8, chronos2.WithCrossLearning(), chronos2.WithBatchSize(2))
	// Then
	if err != nil || len(results) != 2 {
		t.Errorf("ForecastTasks() = %d results, error %v; want 2, nil", len(results), err)
	}
}

// TestModel_ForecastMultivariate_CrossLearningOnSingleTask_HasNoEffect verifies that WithCrossLearning has no effect on a single task.
func TestModel_ForecastMultivariate_CrossLearningOnSingleTask_HasNoEffect(t *testing.T) {
	t.Parallel()

	// Given a single multivariate task forecast without cross-learning.
	model := synthetic(t)
	variates := [][]float64{wave(90, 0), wave(90, 1)}
	plain, err := model.ForecastMultivariate(t.Context(), variates, 12)
	if err != nil {
		t.Fatalf("ForecastMultivariate() error = %v", err)
	}

	// When
	crossLearned, err := model.ForecastMultivariate(t.Context(), variates, 12, chronos2.WithCrossLearning())
	// Then
	if err != nil {
		t.Fatalf("ForecastMultivariate(WithCrossLearning) error = %v", err)
	}
	for i := range variates {
		if !equalBits(forecastBits(t, crossLearned[i]), forecastBits(t, plain[i])) {
			t.Errorf("variate %d: cross-learning changed a single-task forecast", i)
		}
	}
}
