//go:build slow

package parity_test

import (
	"fmt"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	base "github.com/pinkhop/chronos2-go-base"
)

// BenchmarkBaseForecast measures a single-pass base-model forecast at each
// benchmark context length.
func BenchmarkBaseForecast(b *testing.B) {
	model := loadModel(b, base.New)
	for _, contextLength := range benchmarkContextLengths {
		b.Run(fmt.Sprintf("context=%d", contextLength), func(b *testing.B) {
			benchmarkForecast(b, model, contextLength, benchmarkHorizon)
		})
	}
}

// BenchmarkBaseForecastLongHorizon measures a base-model forecast that
// unrolls past one pass.
func BenchmarkBaseForecastLongHorizon(b *testing.B) {
	model := loadModel(b, base.New)
	benchmarkForecast(b, model, benchmarkLongHorizonContext, benchmarkLongHorizon, chronos2.WithLongHorizon())
}

// BenchmarkBaseEmbed measures a base-model embedding of one history.
func BenchmarkBaseEmbed(b *testing.B) {
	benchmarkEmbed(b, loadModel(b, base.New), benchmarkLongHorizonContext)
}
