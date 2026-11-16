package chronos2

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// TestModel_ForecastTasks_InvalidInput_ReturnsSentinel verifies that Model.ForecastTasks, given invalid input, returns sentinel.
func TestModel_ForecastTasks_InvalidInput_ReturnsSentinel(t *testing.T) {
	t.Parallel()

	tests := slices.Concat(invalidValueCases(), invalidStructureCases())
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given
			model, forward, _ := newFakeModel()

			// When
			_, err := model.ForecastTasks(t.Context(), tt.tasks, tt.horizon, tt.options...)

			// Then
			if !errors.Is(err, tt.want) {
				t.Errorf("ForecastTasks() error = %v, want %v", err, tt.want)
			}
			if len(forward.batches) != 0 {
				t.Errorf("engine ran %d passes on invalid input", len(forward.batches))
			}
		})
	}
}

// invalidInputCase is one invalid ForecastTasks call and the sentinel its
// error must match.
type invalidInputCase struct {
	// name labels the subtest.
	name string

	// tasks is the call's task list.
	tasks []Task

	// horizon is the call's horizon.
	horizon int

	// options are the call's options.
	options []ForecastOption

	// want is the sentinel the error must match.
	want error
}

// invalidValueCases returns calls with an invalid horizon or invalid target
// values.
func invalidValueCases() []invalidInputCase {
	nan, inf := math.NaN(), math.Inf(1)
	history := ramp(1, 8)
	return []invalidInputCase{
		{"horizon zero", []Task{{Targets: [][]float64{history}}}, 0, nil, ErrInvalidHorizon},
		{"horizon too long", []Task{{Targets: [][]float64{history}}}, 65, nil, ErrHorizonTooLong},
		{"no targets", []Task{{}}, 4, nil, ErrEmptyHistory},
		{"empty target", []Task{{Targets: [][]float64{{}}}}, 4, nil, ErrEmptyHistory},
		{"all NaN target", []Task{{Targets: [][]float64{{nan, nan}}}}, 4, nil, ErrNoObservations},
		{"infinite target", []Task{{Targets: [][]float64{{1, inf}}}}, 4, nil, ErrInvalidValue},
		{"target beyond float32", []Task{{Targets: [][]float64{{1, 1e300}}}}, 4, nil, ErrInvalidValue},
		{"negative target beyond float32", []Task{{Targets: [][]float64{{1, -math.MaxFloat64}}}}, 4, nil, ErrInvalidValue},
		{"targets differ in length", []Task{{Targets: [][]float64{history, ramp(1, 7)}}}, 4, nil, ErrMismatchedLengths},
	}
}

// invalidStructureCases returns calls with invalid covariates or options.
func invalidStructureCases() []invalidInputCase {
	inf := math.Inf(1)
	history := ramp(1, 8)
	return []invalidInputCase{
		{"past covariate length", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{Past: map[string][]float64{"x": ramp(0, 7)}},
		}}, 4, nil, ErrMismatchedLengths},
		{"future covariate length", []Task{{
			Targets:    [][]float64{history},
			Covariates: Covariates{Past: map[string][]float64{"x": history}, Future: map[string][]float64{"x": {1, 2}}},
		}}, 4, nil, ErrMismatchedLengths},
		{"categorical past length", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{PastCategorical: map[string][]string{"c": {"a"}}},
		}}, 4, nil, ErrMismatchedLengths},
		{"infinite covariate", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{Past: map[string][]float64{"x": {1, 2, 3, 4, 5, 6, 7, inf}}},
		}}, 4, nil, ErrInvalidValue},
		{"past covariate beyond float32", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{Past: map[string][]float64{"x": {1, 2, 3, 4, 5, 6, 7, 4e38}}},
		}}, 4, nil, ErrInvalidValue},
		{"future covariate beyond float32", []Task{{
			Targets:    [][]float64{history},
			Covariates: Covariates{Past: map[string][]float64{"x": history}, Future: map[string][]float64{"x": {1, -1e39, 3, 4}}},
		}}, 4, nil, ErrInvalidValue},
		{"future without past", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{Future: map[string][]float64{"x": {1, 2, 3, 4}}},
		}}, 4, nil, ErrInvalidCovariates},
		{"categorical future without past", []Task{{
			Targets: [][]float64{history}, Covariates: Covariates{FutureCategorical: map[string][]string{"c": {"a", "a", "a", "a"}}},
		}}, 4, nil, ErrInvalidCovariates},
		{"numeric and categorical", []Task{{
			Targets: [][]float64{history},
			Covariates: Covariates{
				Past:            map[string][]float64{"x": history},
				PastCategorical: map[string][]string{"x": {"a", "a", "a", "a", "a", "a", "a", "a"}},
			},
		}}, 4, nil, ErrInvalidCovariates},
		{
			"cross-learning too large",
			[]Task{{Targets: [][]float64{history}}, {Targets: [][]float64{history}}},
			4,
			[]ForecastOption{WithCrossLearning(), WithBatchSize(1)},
			ErrCrossLearningBatchTooLarge,
		},
		{"context length zero", []Task{{Targets: [][]float64{history}}}, 4, []ForecastOption{WithContextLength(0)}, ErrInvalidOption},
		{"context length too long", []Task{{Targets: [][]float64{history}}}, 4, []ForecastOption{WithContextLength(65)}, ErrInvalidOption},
		{"parallelism zero", []Task{{Targets: [][]float64{history}}}, 4, []ForecastOption{WithMaxParallelism(0)}, ErrInvalidOption},
		{"batch size zero", []Task{{Targets: [][]float64{history}}}, 4, []ForecastOption{WithBatchSize(0)}, ErrInvalidOption},
	}
}

// TestModel_ForecastTasks_NaNInFutureCovariate_Accepted verifies that Model.ForecastTasks, given NaN in future covariate, is accepted.
func TestModel_ForecastTasks_NaNInFutureCovariate_Accepted(t *testing.T) {
	t.Parallel()

	// Given an unknown future value, which the model treats as missing.
	model, _, _ := newFakeModel()
	task := Task{
		Targets:    [][]float64{ramp(1, 4)},
		Covariates: Covariates{Past: map[string][]float64{"x": ramp(0, 4)}, Future: map[string][]float64{"x": {1, math.NaN()}}},
	}

	// When
	_, err := model.ForecastTasks(t.Context(), []Task{task}, 2)
	// Then
	if err != nil {
		t.Errorf("ForecastTasks() error = %v, want nil", err)
	}
}

// TestModel_ForecastTasks_ValuesAtFloat32Limits_Accepted verifies that
// Model.ForecastTasks accepts values whose magnitude equals math.MaxFloat32,
// the largest finite float32, in targets and in past and future covariates.
func TestModel_ForecastTasks_ValuesAtFloat32Limits_Accepted(t *testing.T) {
	t.Parallel()

	// Given values of exactly ±math.MaxFloat32
	model, _, _ := newFakeModel()
	task := Task{
		Targets: [][]float64{{1, math.MaxFloat32, 3, -math.MaxFloat32}},
		Covariates: Covariates{
			Past:   map[string][]float64{"x": {math.MaxFloat32, 0, 0, 0}},
			Future: map[string][]float64{"x": {-math.MaxFloat32, 0}},
		},
	}

	// When
	_, err := model.ForecastTasks(t.Context(), []Task{task}, 2)
	// Then
	if err != nil {
		t.Errorf("ForecastTasks() error = %v, want nil", err)
	}
}

// TestModel_Forecast_HorizonBeyondMaxWithLongHorizon_Accepted verifies that Model.Forecast, given horizon beyond max with long horizon, is accepted.
func TestModel_Forecast_HorizonBeyondMaxWithLongHorizon_Accepted(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()

	// When
	_, err := model.Forecast(t.Context(), ramp(1, 8), 65, WithLongHorizon())
	// Then
	if err != nil {
		t.Errorf("Forecast() error = %v, want nil", err)
	}
}

// TestModel_ForecastBatch_BadHistory_NamesHistoryIndex verifies that Model.ForecastBatch, given bad history, names history index.
func TestModel_ForecastBatch_BadHistory_NamesHistoryIndex(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()

	// When
	_, err := model.ForecastBatch(t.Context(), [][]float64{ramp(1, 4), {}}, 2)

	// Then
	if !errors.Is(err, ErrEmptyHistory) || err.Error() != "history 1: chronos2: empty history" {
		t.Errorf("error = %q, want ErrEmptyHistory naming history 1", err)
	}
}
