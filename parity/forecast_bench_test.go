package parity_test

import (
	"fmt"
	"math"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	small "github.com/pinkhop/chronos2-go-small"
)

// benchmarkContextLengths are the history lengths the end-to-end forecast
// benchmarks run at, from a short window to the models' full context.
var benchmarkContextLengths = []int{512, 2048, 8192}

// Benchmark shapes shared by both models' benchmarks.
const (
	// benchmarkHorizon is the single-pass forecast horizon.
	benchmarkHorizon = 64

	// benchmarkLongHorizon exceeds one 1024-step pass, so it unrolls one round.
	benchmarkLongHorizon = 1100

	// benchmarkLongHorizonContext is the history length of the long-horizon
	// and embedding benchmarks.
	benchmarkLongHorizonContext = 2048
)

// BenchmarkSmallForecast measures a single-pass small-model forecast at each
// benchmark context length.
func BenchmarkSmallForecast(b *testing.B) {
	model := loadModel(b, small.New)
	for _, contextLength := range benchmarkContextLengths {
		b.Run(fmt.Sprintf("context=%d", contextLength), func(b *testing.B) {
			benchmarkForecast(b, model, contextLength, benchmarkHorizon)
		})
	}
}

// BenchmarkSmallForecastLongHorizon measures a small-model forecast that
// unrolls past one pass.
func BenchmarkSmallForecastLongHorizon(b *testing.B) {
	model := loadModel(b, small.New)
	benchmarkForecast(b, model, benchmarkLongHorizonContext, benchmarkLongHorizon, chronos2.WithLongHorizon())
}

// BenchmarkSmallEmbed measures a small-model embedding of one history.
func BenchmarkSmallEmbed(b *testing.B) {
	benchmarkEmbed(b, loadModel(b, small.New), benchmarkLongHorizonContext)
}

// loadModel decodes a model with newModel, failing b on error. It runs before
// the timed loop, so decoding is excluded from the measurement.
func loadModel(b *testing.B, newModel func() (*chronos2.Model, error)) *chronos2.Model {
	b.Helper()
	model, err := newModel()
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	return model
}

// benchmarkForecast times m.Forecast over a synthetic history of
// contextLength steps and reports the throughput in series per second.
func benchmarkForecast(b *testing.B, m *chronos2.Model, contextLength, horizon int, opts ...chronos2.ForecastOption) {
	history := syntheticHistory(contextLength)
	ctx := b.Context()
	for b.Loop() {
		if _, err := m.Forecast(ctx, history, horizon, opts...); err != nil {
			b.Fatalf("Forecast: %v", err)
		}
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "series/s")
}

// benchmarkEmbed times m.Embed over a synthetic history of contextLength steps
// and reports the throughput in series per second.
func benchmarkEmbed(b *testing.B, m *chronos2.Model, contextLength int) {
	history := syntheticHistory(contextLength)
	ctx := b.Context()
	for b.Loop() {
		if _, err := m.Embed(ctx, history); err != nil {
			b.Fatalf("Embed: %v", err)
		}
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "series/s")
}

// syntheticHistory returns a deterministic history of n steps: a level of 100
// with a gentle upward trend, a 24-step daily cycle, and a 168-step weekly
// cycle, like hourly load data.
func syntheticHistory(n int) []float64 {
	const (
		level        = 100.0
		trendPerStep = 0.01
		dailyPeriod  = 24.0
		dailyAmount  = 10.0
		weeklyPeriod = 168.0
		weeklyAmount = 4.0
	)
	history := make([]float64, n)
	for i := range history {
		step := float64(i)
		history[i] = level + trendPerStep*step +
			dailyAmount*math.Sin(2*math.Pi*step/dailyPeriod) +
			weeklyAmount*math.Sin(2*math.Pi*step/weeklyPeriod)
	}
	return history
}
