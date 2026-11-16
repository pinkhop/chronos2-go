package testmodel

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Scenario is one pipeline-level reference fixture: the inputs and options of
// a public-API call and the expected outputs, read from <name>.json beside
// <name>.safetensors.
type Scenario struct {
	// Name identifies the scenario, such as "univariate_short".
	Name string `json:"scenario"`
	// Method names the public method the scenario exercises: Forecast,
	// ForecastBatch, ForecastMultivariate, ForecastWithCovariates,
	// ForecastTasks, or Embed.
	Method string `json:"method"`
	// Horizon is the forecast horizon in steps; zero for Embed.
	Horizon int `json:"horizon"`
	// Options holds the call's options.
	Options ScenarioOptions `json:"options"`
	// Slow marks scenarios that belong to the slow test tier.
	Slow bool `json:"slow"`
	// Quantiles lists the model's native quantile levels, ascending.
	Quantiles []float64 `json:"quantiles"`
	// Tasks holds the call's inputs, one entry per task; Forecast, Embed,
	// and the other single-input methods have one.
	Tasks []ScenarioTask `json:"tasks"`
}

// ScenarioOptions holds the options a scenario passes; zero values mean the
// option is absent.
type ScenarioOptions struct {
	// ContextLength is the WithContextLength value, or zero.
	ContextLength int `json:"context_length"`
	// BatchSize is the WithBatchSize value, or zero.
	BatchSize int `json:"batch_size"`
	// CrossLearning reports whether WithCrossLearning is passed.
	CrossLearning bool `json:"cross_learning"`
	// LongHorizon reports whether WithLongHorizon is passed.
	LongHorizon bool `json:"long_horizon"`
}

// ScenarioTask is one task's inputs. Numeric series use NaN for missing
// values; categorical series use "" for the missing category.
type ScenarioTask struct {
	// Targets holds the series to forecast.
	Targets []Series `json:"targets"`
	// Past holds numeric covariates over the history.
	Past map[string]Series `json:"past"`
	// Future holds numeric covariates over the horizon.
	Future map[string]Series `json:"future"`
	// PastCategorical holds categorical covariates over the history.
	PastCategorical map[string][]string `json:"past_categorical"`
	// FutureCategorical holds categorical covariates over the horizon.
	FutureCategorical map[string][]string `json:"future_categorical"`
}

// Series is a numeric series whose JSON form writes NaN as null.
type Series []float64

// UnmarshalJSON decodes a JSON array of numbers and nulls, mapping null to
// NaN.
func (s *Series) UnmarshalJSON(data []byte) error {
	var raw []*float64
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	values := make(Series, len(raw))
	for i, v := range raw {
		if v == nil {
			values[i] = math.NaN()
			continue
		}
		values[i] = *v
	}
	*s = values
	return nil
}

// ReadScenario reads a scenario manifest such as
// testdata/synthetic/scenarios/univariate_short.json and the expected outputs
// in the matching .safetensors file. Forecast outputs are named task_000,
// task_001, and so on, each [targets, quantiles, horizon]; Embed outputs are
// embedding ([patches + 2, width]), loc, and scale. It fails the test when
// either file cannot be read.
func ReadScenario(tb testing.TB, manifestPath string) (Scenario, map[string]Tensor) {
	tb.Helper()
	// #nosec G304 -- tests pass fixed repository fixture paths; no untrusted input.
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		tb.Fatalf("read scenario %s: %v", manifestPath, err)
	}
	var scenario Scenario
	if err := json.Unmarshal(data, &scenario); err != nil {
		tb.Fatalf("decode scenario %s: %v", manifestPath, err)
	}
	outputs := ReadTensors(tb, manifestPath[:len(manifestPath)-len(".json")]+".safetensors")
	return scenario, outputs
}
