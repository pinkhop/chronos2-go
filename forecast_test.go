package chronos2

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// newTestForecast returns a two-step forecast over the small model's levels
// whose line at level index i is {i, 10 + i}.
func newTestForecast() Forecast {
	values := make([]float32, 0, 2*len(testLevels))
	for level := range testLevels {
		values = append(values, float32(level), float32(10+level))
	}
	return newForecast(testLevels, values, 2)
}

// TestForecast_Median_SmallModel_ReturnsNativeLine verifies that Forecast.Median, given small model, returns native line.
func TestForecast_Median_SmallModel_ReturnsNativeLine(t *testing.T) {
	t.Parallel()

	// Given level 0.5 at index 6.
	forecast := newTestForecast()

	// When
	median := forecast.Median()

	// Then
	if !slices.Equal(median, []float64{6, 16}) {
		t.Errorf("Median() = %v, want [6 16]", median)
	}
}

// TestForecast_Percentile_NativeLevel_ReturnsExactLine verifies that Forecast.Percentile, given native level, returns exact line.
func TestForecast_Percentile_NativeLevel_ReturnsExactLine(t *testing.T) {
	t.Parallel()

	// Given
	forecast := newTestForecast()

	// When
	line, err := forecast.Percentile(95)
	// Then
	if err != nil || !slices.Equal(line, []float64{11, 21}) {
		t.Errorf("Percentile(95) = %v, %v; want [11 21], nil", line, err)
	}
}

// TestForecast_Percentile_BetweenLevels_Interpolates verifies that Forecast.Percentile interpolates between native levels.
func TestForecast_Percentile_BetweenLevels_Interpolates(t *testing.T) {
	t.Parallel()

	// Given levels 0.2 (index 3) and 0.3 (index 4).
	forecast := newTestForecast()

	// When
	line, err := forecast.Percentile(25)
	// Then
	if err != nil || len(line) != 2 || math.Abs(line[0]-3.5) > 1e-6 || math.Abs(line[1]-13.5) > 1e-6 {
		t.Errorf("Percentile(25) = %v, %v; want [3.5 13.5], nil", line, err)
	}
}

// TestForecast_Percentile_OutOfRange_ReturnsSentinel verifies that Forecast.Percentile, given an out-of-range percent, returns the sentinel.
func TestForecast_Percentile_OutOfRange_ReturnsSentinel(t *testing.T) {
	t.Parallel()

	for _, p := range []float64{0.5, 99.5, math.NaN(), -1} {
		t.Run("", func(t *testing.T) {
			t.Parallel()

			// Given
			forecast := newTestForecast()

			// When
			_, err := forecast.Percentile(p)

			// Then
			if !errors.Is(err, ErrPercentileOutOfRange) {
				t.Errorf("Percentile(%v) error = %v, want ErrPercentileOutOfRange", p, err)
			}
		})
	}
}

// TestForecast_Percentile15_SmallModel_ReturnsNil verifies that Forecast.Percentile15, given small model, returns nil.
func TestForecast_Percentile15_SmallModel_ReturnsNil(t *testing.T) {
	t.Parallel()

	// Given
	forecast := newTestForecast()

	// When
	line := forecast.Percentile15()

	// Then
	if line != nil {
		t.Errorf("Percentile15() = %v, want nil", line)
	}
}

// TestForecast_Percentile99_SmallModel_ReturnsNativeLine verifies that Forecast.Percentile99, given small model, returns native line.
func TestForecast_Percentile99_SmallModel_ReturnsNativeLine(t *testing.T) {
	t.Parallel()

	// Given
	forecast := newTestForecast()

	// When
	line := forecast.Percentile99()

	// Then
	if !slices.Equal(line, []float64{12, 22}) {
		t.Errorf("Percentile99() = %v, want [12 22]", line)
	}
}

// TestForecast_Median_CallerModifiesResult_ForecastUnchanged verifies that Forecast.Median, given caller modifies result, leaves forecast unchanged.
func TestForecast_Median_CallerModifiesResult_ForecastUnchanged(t *testing.T) {
	t.Parallel()

	// Given
	forecast := newTestForecast()
	first := forecast.Median()
	first[0] = 999

	// When
	second := forecast.Median()

	// Then
	if second[0] != 6 {
		t.Errorf("Median()[0] = %v after modifying an earlier result, want 6", second[0])
	}
}

// TestForecast_NativePercentiles_SmallModel_ReturnsRoundedPercents verifies that Forecast.NativePercentiles, given small model, returns rounded percents.
func TestForecast_NativePercentiles_SmallModel_ReturnsRoundedPercents(t *testing.T) {
	t.Parallel()

	// Given
	forecast := newTestForecast()

	// When
	percents := forecast.NativePercentiles()

	// Then
	if want := []float64{1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99}; !slices.Equal(percents, want) {
		t.Errorf("NativePercentiles() = %v, want %v", percents, want)
	}
}

// TestForecast_ZeroValue_AccessorsReturnEmpty verifies that the zero Forecast's accessors return empty results.
func TestForecast_ZeroValue_AccessorsReturnEmpty(t *testing.T) {
	t.Parallel()

	// Given
	var forecast Forecast

	// When
	line, err := forecast.Percentile(50)

	// Then
	if line != nil || err != nil || forecast.Horizon() != 0 {
		t.Errorf("zero Forecast Percentile(50) = %v, %v; Horizon = %d", line, err, forecast.Horizon())
	}
}

// TestModel_Accessors_FakeSpec_ReportCapabilities verifies that the Model capability accessors report the spec's values.
func TestModel_Accessors_FakeSpec_ReportCapabilities(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()

	// When
	name, context, horizon, width, percents := model.Name(), model.ContextLength(), model.MaxHorizon(), model.EmbeddingDimension(), model.NativePercentiles()

	// Then
	if name != "fake" || context != 64 || horizon != 64 || width != 4 || len(percents) != 13 || percents[6] != 50 {
		t.Errorf("accessors = %q %d %d %d %v", name, context, horizon, width, percents)
	}
}
