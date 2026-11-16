package horizon_test

import (
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/horizon"
	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// unrolledDeciles are the levels whose paths the unroll follows.
var unrolledDeciles = []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

// firstDecileIndex is the index of the 0.1 level in smallLevels; decile path
// n reads the quantile at firstDecileIndex + n.
const firstDecileIndex = 2

// TestForecast_LongHorizon_RoundsCoverHorizonWithGrowingNinePathContexts verifies that Forecast, given long horizon, rounds cover horizon with growing nine path contexts.
func TestForecast_LongHorizon_RoundsCoverHorizonWithGrowingNinePathContexts(t *testing.T) {
	t.Parallel()

	// Given three rows in two groups, a four-step context, a 40-step horizon,
	// one patch per pass, and a 20-step full context window
	batch := newBatch([][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, []int{0, 0, 1}, 48, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	result, err := horizon.Forecast(t.Context(), fake, batch, 40, unrollConfig())
	// Then three one-patch passes run, the later two over nine paths per row
	// with contexts grown by 16 steps and trimmed to the full 20-step window
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.OutputPatches }); !slices.Equal(got, []int{1, 1, 1}) {
		t.Errorf("OutputPatches per pass = %v, want [1 1 1]", got)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.Rows }); !slices.Equal(got, []int{3, 27, 27}) {
		t.Errorf("Rows per pass = %v, want [3 27 27]", got)
	}
	if got := field(fake.batches, func(b networkBatch) int { return b.ContextLength }); !slices.Equal(got, []int{4, 20, 20}) {
		t.Errorf("ContextLength per pass = %v, want [4 20 20]", got)
	}
	if len(result) != 3*len(smallLevels)*40 {
		t.Errorf("len(result) = %d, want %d", len(result), 3*len(smallLevels)*40)
	}
}

// TestForecast_LongHorizon_PathGroupsAreGroupTimesNinePlusPath verifies that Forecast gives each long-horizon path the group g × 9 + path.
func TestForecast_LongHorizon_PathGroupsAreGroupTimesNinePlusPath(t *testing.T) {
	t.Parallel()

	// Given rows in groups 0, 0, and 1 and a two-pass horizon
	batch := newBatch([][]float32{{1}, {2}, {3}}, []int{0, 0, 1}, 32, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 32, unrollConfig())
	// Then each row expands into nine consecutive paths with group g×9 + path
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	paths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8}
	shifted := []int{9, 10, 11, 12, 13, 14, 15, 16, 17}
	want := slices.Concat(paths, paths, shifted)
	if got := fake.batches[1].GroupIDs; !slices.Equal(got, want) {
		t.Errorf("GroupIDs = %v, want %v", got, want)
	}
}

// TestForecast_LongHorizon_AppendsEachPathsDecileOfPreviousForecast verifies that Forecast appends each path's decile of the previous forecast to its context.
func TestForecast_LongHorizon_AppendsEachPathsDecileOfPreviousForecast(t *testing.T) {
	t.Parallel()

	// Given two four-step rows with no known future and a two-pass horizon
	batch := newBatch([][]float32{{1, 2, 3, 4}, {nan, 6, 7, 8}}, []int{0, 1}, 32, nil)
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 32, unrollConfig())
	// Then path n of each row holds the row's context followed by the first
	// pass's prediction at decile n+1, an exact native level
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	second := fake.batches[1]
	for r, context := range [][]float32{{1, 2, 3, 4}, {nan, 6, 7, 8}} {
		for n := range unrolledDeciles {
			want := slices.Clone(context)
			for s := range 16 {
				want = append(want, gridValue(0, r, firstDecileIndex+n, s))
			}
			if got := rowOf(second.Context, r*9+n, second.ContextLength); !equalFloats(got, want) {
				t.Errorf("row %d path %d context = %v, want %v", r, n, got, want)
			}
		}
	}
}

// TestForecast_LongHorizon_KnownFutureReplacesPredictionsAndShiftsWindow verifies that Forecast, given long horizon, known future replaces predictions and shifts window.
func TestForecast_LongHorizon_KnownFutureReplacesPredictionsAndShiftsWindow(t *testing.T) {
	t.Parallel()

	// Given a known-future row with values at steps 0, 2, 16, and 32 only and
	// a three-pass horizon
	known := nanFloats(48)
	known[0], known[2], known[16], known[32] = 70, 72, 86, 102
	batch := newBatch([][]float32{{1, 2, 3, 4}}, []int{0}, 48, map[int][]float32{0: known})
	fake := &fakeForwarder{levels: smallLevels, value: gridValue}

	// When the batch is forecast
	_, err := horizon.Forecast(t.Context(), fake, batch, 40, unrollConfig())
	// Then the second pass's contexts carry the actual values where known and
	// the predicted decile elsewhere, and each pass's future window starts
	// where its predictions start
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	second := fake.batches[1]
	path := 4
	want := []float32{1, 2, 3, 4}
	for s := range 16 {
		want = append(want, gridValue(0, 0, firstDecileIndex+path, s))
	}
	want[4], want[6] = 70, 72
	if got := rowOf(second.Context, path, second.ContextLength); !equalFloats(got, want) {
		t.Errorf("path %d context = %v, want %v", path, got, want)
	}
	for pass, start := range []int{0, 16, 32} {
		rows := fake.batches[pass].Rows
		wantFuture := slices.Repeat(known[start:start+16], rows)
		if got := fake.batches[pass].FutureCovariates; !equalFloats(got, wantFuture) {
			t.Errorf("pass %d future = %v, want %v", pass, got, wantFuture)
		}
	}
}

// TestForecast_LongHorizon_ReducesNinePathsWithOuterProductLevelMass verifies that Forecast, given long horizon, reduces nine paths with outer product level mass.
func TestForecast_LongHorizon_ReducesNinePathsWithOuterProductLevelMass(t *testing.T) {
	t.Parallel()

	// Given a second pass whose predictions vary by path, quantile, and step
	batch := newBatch([][]float32{{1, 2}}, []int{0}, 32, nil)
	value := func(pass, row, q, step int) float32 {
		return float32((row*31+q*17)%23) + 0.5*float32(step) + float32(pass)
	}
	fake := &fakeForwarder{levels: smallLevels, value: value}

	// When a two-pass horizon is forecast
	result, err := horizon.Forecast(t.Context(), fake, batch, 32, unrollConfig())
	// Then each later step is the weighted quantile of the nine paths' native
	// predictions, weighted by the outer product of decile and native masses
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	quantiles := len(smallLevels)
	weights := outerMass(quantile.LevelMass(unrolledDeciles), quantile.LevelMass(smallLevels))
	for s := range 16 {
		samples := make([]float32, 0, 9*quantiles)
		for path := range 9 {
			for q := range quantiles {
				samples = append(samples, value(1, path, q, s))
			}
		}
		want := quantile.WeightedQuantile(smallLevels, weights, samples)
		for q := range quantiles {
			if got := result[q*32+16+s]; got != want[q] {
				t.Errorf("step %d quantile %d = %v, want %v", 16+s, q, got, want[q])
			}
		}
	}
}

// outerMass returns the outer product of two mass vectors, flattened
// path-major, with each product computed in float32 as the reference does.
func outerMass(paths, levels []float64) []float64 {
	weights := make([]float64, 0, len(paths)*len(levels))
	for _, p := range paths {
		for _, l := range levels {
			weights = append(weights, float64(float32(p)*float32(l)))
		}
	}
	return weights
}
