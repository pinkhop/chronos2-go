package chronos2_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// sharedModel loads the synthetic checkpoint once per test binary. A *Model is
// immutable and safe for concurrent use, so tests may share it freely; doing
// so also exercises that guarantee.
var sharedModel = sync.OnceValues(func() (*chronos2.Model, error) {
	dir := testmodel.SyntheticDir()
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	shard, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	return chronos2.Load(chronos2.Checkpoint{Name: "synthetic", Config: config, Shards: [][]byte{shard}})
})

// synthetic returns the shared synthetic model, failing the test when it
// cannot be loaded.
func synthetic(tb testing.TB) *chronos2.Model {
	tb.Helper()
	model, err := sharedModel()
	if err != nil {
		tb.Fatalf("load synthetic model: %v", err)
	}
	return model
}

// publicSentinels lists every error a method may return: the sentinels
// errors.go declares, plus the context errors cancellation surfaces.
var publicSentinels = []error{
	chronos2.ErrEmptyHistory, chronos2.ErrNoObservations, chronos2.ErrInvalidValue,
	chronos2.ErrInvalidHorizon, chronos2.ErrHorizonTooLong, chronos2.ErrMismatchedLengths,
	chronos2.ErrInvalidCovariates, chronos2.ErrInvalidOption, chronos2.ErrCrossLearningBatchTooLarge,
	chronos2.ErrPercentileOutOfRange, chronos2.ErrInvalidCheckpoint,
	context.Canceled, context.DeadlineExceeded,
}

// matchesSentinel reports whether err matches one of publicSentinels.
func matchesSentinel(err error) bool {
	for _, sentinel := range publicSentinels {
		if errors.Is(err, sentinel) {
			return true
		}
	}
	return false
}

// forecastBits returns every native line of forecast as raw float64 bits,
// so results can be compared exactly. It fails the test if a native line is
// unavailable.
func forecastBits(tb testing.TB, forecast chronos2.Forecast) [][]uint64 {
	tb.Helper()
	percents := forecast.NativePercentiles()
	lines := make([][]uint64, len(percents))
	for i, percent := range percents {
		line, err := forecast.Percentile(percent)
		if err != nil {
			tb.Fatalf("Percentile(%v) error = %v", percent, err)
		}
		lines[i] = make([]uint64, len(line))
		for step, value := range line {
			lines[i][step] = math.Float64bits(value)
		}
	}
	return lines
}

// equalBits reports whether two forecastBits results are identical.
func equalBits(a, b [][]uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// wave returns n steps of a seasonal series offset by phase, with every
// seventh step missing, so histories differ and exercise masking.
func wave(n int, phase float64) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = 10 + 3*math.Sin(float64(i)/4+phase) + float64(i)/50
		if i%7 == 3 {
			values[i] = math.NaN()
		}
	}
	return values
}
