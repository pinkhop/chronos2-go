package chronos2_test

import (
	"math"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// TestModel_ForecastWithCovariates_KnownFutureCovariateWithAllMissingPast_ReturnsFiniteForecast
// verifies that a known-future covariate whose past is entirely missing, so
// its normalization falls back to location 0 and scale 1, still yields a
// finite forecast of the requested length.
func TestModel_ForecastWithCovariates_KnownFutureCovariateWithAllMissingPast_ReturnsFiniteForecast(t *testing.T) {
	t.Parallel()

	// Given a target and a known-future covariate with an all-NaN past and a
	// horizon that is not a multiple of 16
	const horizon = 20
	model := synthetic(t)
	target := make([]float64, 48)
	missingPast := make([]float64, len(target))
	for i := range target {
		target[i] = 10 + math.Sin(float64(i)/4)
		missingPast[i] = math.NaN()
	}
	future := make([]float64, horizon)
	for i := range future {
		future[i] = float64(i % 2)
	}
	task := chronos2.CovariateTask{
		Target: target,
		Covariates: chronos2.Covariates{
			Past:   map[string][]float64{"promotion": missingPast},
			Future: map[string][]float64{"promotion": future},
		},
	}

	// When the target is forecast
	forecast, err := model.ForecastWithCovariates(t.Context(), task, horizon)
	// Then the call succeeds with a finite median of horizon steps
	if err != nil {
		t.Fatalf("ForecastWithCovariates: %v", err)
	}
	median := forecast.Median()
	if len(median) != horizon {
		t.Fatalf("median has %d steps, want %d", len(median), horizon)
	}
	for step, value := range median {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Errorf("median[%d] = %v, want a finite value", step, value)
		}
	}
}
