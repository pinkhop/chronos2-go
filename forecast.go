package chronos2

import (
	"fmt"
	"math"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// Percentile bounds accepted by Forecast.Percentile.
const (
	// minPercentile is the lowest percent Forecast.Percentile accepts.
	minPercentile = 1

	// maxPercentile is the highest percent Forecast.Percentile accepts.
	maxPercentile = 99

	// medianPercent is the percentile of the median line.
	medianPercent = 50
)

// Forecast is the result of forecasting one target, in the target's units.
// It holds one line per native percentile of the producing model (see
// [Model.NativePercentiles]): the p-th percentile line holds, for each future
// step, the value the model expects the actual value to fall below with
// probability p%. Every line is Horizon steps long, nearest step first.
//
// A Forecast is immutable and safe for concurrent use; every accessor returns
// a newly allocated slice the caller may modify. The zero Forecast has no
// lines and a horizon of 0.
type Forecast struct {
	// levels lists the native quantile levels in (0, 1), ascending. It is
	// shared with the producing Model and never modified.
	levels []float64

	// lines holds one line per level, each horizon steps long. It is owned by
	// this Forecast and never modified.
	lines [][]float64
}

// newForecast builds a Forecast for the given levels from values, which
// holds len(levels) × horizon float32 predictions in level, then step order.
func newForecast(levels []float64, values []float32, horizon int) Forecast {
	lines := make([][]float64, len(levels))
	for level := range lines {
		line := make([]float64, horizon)
		for step, value := range values[level*horizon : (level+1)*horizon] {
			line[step] = float64(value)
		}
		lines[level] = line
	}
	return Forecast{levels: levels, lines: lines}
}

// Horizon returns the number of forecast steps in every line.
func (f Forecast) Horizon() int {
	if len(f.lines) == 0 {
		return 0
	}
	return len(f.lines[0])
}

// NativePercentiles returns, in ascending order, the percentile levels this
// forecast holds lines for, matching the producing model's
// [Model.NativePercentiles]. The caller owns the returned slice.
func (f Forecast) NativePercentiles() []float64 { return levelsToPercents(f.levels) }

// Median returns the 50th-percentile line, the conventional point forecast:
// the actual value is equally likely to fall above or below it. Both
// published models predict it natively. The zero Forecast returns nil.
func (f Forecast) Median() []float64 {
	if line := f.native(medianPercent); line != nil {
		return line
	}
	line, err := f.Percentile(medianPercent)
	if err != nil {
		// Unreachable: 50 always lies within [1, 99].
		return nil
	}
	return line
}

// Percentile returns the line at percent p, which may be any value from 1 to
// 99 inclusive, such as 92.5. A native percentile returns its line exactly;
// any other interpolates linearly, step by step, between the lines of the two
// nearest native percentiles. A p outside the range, or NaN, returns an error
// wrapping [ErrPercentileOutOfRange]. The zero Forecast returns nil and no
// error.
func (f Forecast) Percentile(p float64) ([]float64, error) {
	if !(p >= minPercentile && p <= maxPercentile) {
		return nil, fmt.Errorf("%w: got %v", ErrPercentileOutOfRange, p)
	}
	if len(f.lines) == 0 {
		return nil, nil
	}
	if line := f.native(p); line != nil {
		return line, nil
	}
	interpolated := quantile.Interpolate([]float64{p / percentScale}, f.levels, f.lines)
	return interpolated[0], nil
}

// native returns a copy of the line at percent p when p is a native level,
// or nil otherwise.
func (f Forecast) native(p float64) []float64 {
	for i, level := range f.levels {
		if roundPercent(level*percentScale) == p {
			return slices.Clone(f.lines[i])
		}
	}
	return nil
}

// roundPercent rounds a percent to percentRounding, so a level such as 0.07
// reports and matches as exactly 7.
func roundPercent(percent float64) float64 {
	return math.Round(percent*percentRounding) / percentRounding
}
