package chronos2_test

import (
	"math"
	"sync"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// concurrentCallers is the number of goroutines that call one Model at once.
const concurrentCallers = 16

// TestModel_ConcurrentCalls_MatchSequentialResults verifies that concurrent calls on one Model match the same calls made sequentially.
func TestModel_ConcurrentCalls_MatchSequentialResults(t *testing.T) {
	t.Parallel()

	// Given sequential reference results for each caller's input.
	model := synthetic(t)
	want := make([]callResults, concurrentCallers)
	for caller := range concurrentCallers {
		want[caller] = callAll(t, model, caller)
	}

	// When every caller runs the same calls concurrently.
	got := make([]callResults, concurrentCallers)
	var wg sync.WaitGroup
	for caller := range concurrentCallers {
		wg.Go(func() { got[caller] = callAll(t, model, caller) })
	}
	wg.Wait()

	// Then
	for caller := range concurrentCallers {
		if !equalBits(got[caller].single, want[caller].single) {
			t.Errorf("caller %d: concurrent Forecast differs from sequential", caller)
		}
		for i := range want[caller].batch {
			if !equalBits(got[caller].batch[i], want[caller].batch[i]) {
				t.Errorf("caller %d: concurrent ForecastBatch[%d] differs from sequential", caller, i)
			}
		}
		if !equalStates(got[caller].states, want[caller].states) {
			t.Errorf("caller %d: concurrent Embed differs from sequential", caller)
		}
	}
}

// callAll runs Forecast, ForecastBatch, and Embed on inputs derived from
// caller, mutating every returned slice afterward to prove the results own
// their memory, and returns the results captured before mutation as
// (Forecast bits, ForecastBatch bits, Embed states).
func callAll(t *testing.T, model *chronos2.Model, caller int) callResults {
	t.Helper()
	phase := float64(caller)
	var result callResults
	forecast, err := model.Forecast(t.Context(), wave(70+caller, phase), 20, chronos2.WithMaxParallelism(2))
	if err != nil {
		t.Errorf("Forecast() error = %v", err)
		return result
	}
	result.single = forecastBits(t, forecast)
	scribble(forecast.Median())
	forecasts, err := model.ForecastBatch(t.Context(), [][]float64{wave(40, phase), wave(90, phase+1)}, 18, chronos2.WithBatchSize(1))
	if err != nil {
		t.Errorf("ForecastBatch() error = %v", err)
		return result
	}
	for _, item := range forecasts {
		result.batch = append(result.batch, forecastBits(t, item))
	}
	embedding, err := model.Embed(t.Context(), wave(50+caller, phase))
	if err != nil {
		t.Errorf("Embed() error = %v", err)
		return result
	}
	result.states = embedding.States()
	scribble32(embedding.Summary())
	return result
}

// callResults holds one caller's results, captured before mutation.
type callResults struct {
	// single holds the Forecast result's bits.
	single [][]uint64

	// batch holds each ForecastBatch result's bits.
	batch [][][]uint64

	// states holds the Embed result's states.
	states [][]float32
}

// TestForecast_AccessorResults_ModifyingThemLeavesForecastUnchanged verifies that modifying slices returned by Forecast accessors leaves the Forecast unchanged.
func TestForecast_AccessorResults_ModifyingThemLeavesForecastUnchanged(t *testing.T) {
	t.Parallel()

	// Given a forecast whose every accessor result has been overwritten.
	model := synthetic(t)
	forecast, err := model.Forecast(t.Context(), wave(64, 0), 16)
	if err != nil {
		t.Fatalf("Forecast() error = %v", err)
	}
	before := forecastBits(t, forecast)
	interpolated, err := forecast.Percentile(42.5)
	if err != nil {
		t.Fatalf("Percentile(42.5) error = %v", err)
	}
	scribble(forecast.Median())
	scribble(forecast.Percentile10())
	scribble(forecast.Percentile99())
	scribble(interpolated)
	scribble(forecast.NativePercentiles())
	scribble(model.NativePercentiles())

	// When
	after := forecastBits(t, forecast)

	// Then
	if !equalBits(after, before) {
		t.Errorf("forecast changed after its accessor results were modified")
	}
}

// TestEmbedding_AccessorResults_ModifyingThemLeavesEmbeddingUnchanged verifies that modifying slices returned by Embedding accessors leaves the Embedding unchanged.
func TestEmbedding_AccessorResults_ModifyingThemLeavesEmbeddingUnchanged(t *testing.T) {
	t.Parallel()

	// Given an embedding whose every accessor result has been overwritten.
	model := synthetic(t)
	embedding, err := model.Embed(t.Context(), wave(64, 0))
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	before := embedding.States()
	scribble32(embedding.Summary())
	for _, patch := range embedding.Patches() {
		scribble32(patch)
	}
	for _, state := range embedding.States() {
		scribble32(state)
	}

	// When
	after := embedding.States()

	// Then
	if !equalStates(after, before) {
		t.Errorf("embedding changed after its accessor results were modified")
	}
}

// TestModel_MaxParallelism_OneAndEight_ProduceIdenticalResults verifies that forecasts with WithMaxParallelism 1 and 8 are identical.
func TestModel_MaxParallelism_OneAndEight_ProduceIdenticalResults(t *testing.T) {
	t.Parallel()

	// Given a call large enough to split across goroutines.
	model := synthetic(t)
	histories := [][]float64{wave(300, 0), wave(250, 1), wave(64, 2), wave(17, 3)}
	sequential, err := model.ForecastBatch(t.Context(), histories, 40, chronos2.WithMaxParallelism(1))
	if err != nil {
		t.Fatalf("ForecastBatch(parallelism 1) error = %v", err)
	}

	// When
	parallel, err := model.ForecastBatch(t.Context(), histories, 40, chronos2.WithMaxParallelism(8))
	// Then
	if err != nil {
		t.Fatalf("ForecastBatch(parallelism 8) error = %v", err)
	}
	for i := range histories {
		if !equalBits(forecastBits(t, parallel[i]), forecastBits(t, sequential[i])) {
			t.Errorf("history %d: parallelism 8 differs from parallelism 1", i)
		}
	}
}

// TestModel_EmbedMaxParallelism_OneAndEight_ProduceIdenticalStates verifies that embeddings with WithMaxParallelism 1 and 8 are identical.
func TestModel_EmbedMaxParallelism_OneAndEight_ProduceIdenticalStates(t *testing.T) {
	t.Parallel()

	// Given
	model := synthetic(t)
	sequential, err := model.Embed(t.Context(), wave(500, 0), chronos2.WithMaxParallelism(1))
	if err != nil {
		t.Fatalf("Embed(parallelism 1) error = %v", err)
	}

	// When
	parallel, err := model.Embed(t.Context(), wave(500, 0), chronos2.WithMaxParallelism(8))
	// Then
	if err != nil {
		t.Fatalf("Embed(parallelism 8) error = %v", err)
	}
	if !equalStates(parallel.States(), sequential.States()) {
		t.Errorf("parallelism 8 states differ from parallelism 1")
	}
}

// scribble overwrites values with NaN.
func scribble(values []float64) {
	for i := range values {
		values[i] = math.NaN()
	}
}

// scribble32 overwrites values with NaN.
func scribble32(values []float32) {
	for i := range values {
		values[i] = float32(math.NaN())
	}
}

// equalStates reports whether two state lists are bitwise identical.
func equalStates(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				return false
			}
		}
	}
	return true
}
