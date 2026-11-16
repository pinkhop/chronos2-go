package chronos2_test

import (
	"fmt"
	"math"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// Bounds that keep each fuzz iteration fast on the synthetic model while
// still crossing the interesting thresholds: patch boundaries, the
// single-pass horizon of 1024 steps, and the long-horizon unroll.
const (
	fuzzMaxHistory = 600
	fuzzMaxHorizon = 1200
	fuzzMaxTargets = 3

	// noTargetsByte makes fuzzTask produce a task with no targets.
	noTargetsByte = 0xFF
)

// fuzzBytes decodes fuzz input into arbitrary inputs, yielding zeros once it
// is exhausted so every input decodes.
type fuzzBytes struct {
	// data is the undecoded remainder.
	data []byte
}

// beyondFloat32 is a finite float64 too large to represent in float32, which
// validation must reject like an infinity.
const beyondFloat32 = 1e300

// next returns the next byte, or 0 once data is exhausted.
func (f *fuzzBytes) next() byte {
	if len(f.data) == 0 {
		return 0
	}
	value := f.data[0]
	f.data = f.data[1:]
	return value
}

// intn returns a value in [0, n) built from two bytes.
func (f *fuzzBytes) intn(n int) int {
	return (int(f.next())<<8 | int(f.next())) % n
}

// value returns a series value: mostly small and finite, sometimes NaN, an
// infinity, or a finite value beyond the float32 range.
func (f *fuzzBytes) value() float64 {
	switch b := f.next(); b {
	case 0xFF:
		return math.NaN()
	case 0xFE:
		return math.Inf(1)
	case 0xFD:
		return math.Inf(-1)
	case 0xFC:
		return beyondFloat32
	case 0xFB:
		return -beyondFloat32
	default:
		return (float64(b) - 128) * 0.37
	}
}

// series returns n values.
func (f *fuzzBytes) series(n int) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = f.value()
	}
	return values
}

// category returns a category name from a small set that includes the
// missing category "".
func (f *fuzzBytes) category() string {
	return [...]string{"", "a", "b", "c"}[f.next()%4]
}

// lengthNear returns length most of the time and a different length
// otherwise, so mismatched lengths are exercised.
func (f *fuzzBytes) lengthNear(length int) int {
	if f.next()%8 == 0 {
		return f.intn(fuzzMaxHistory)
	}
	return length
}

// options decodes a set of forecasting options, some out of range.
func (f *fuzzBytes) options() []chronos2.ForecastOption {
	flags := f.next()
	var options []chronos2.ForecastOption
	if flags&1 != 0 {
		options = append(options, chronos2.WithContextLength(f.intn(fuzzMaxHistory)-2))
	}
	if flags&2 != 0 {
		options = append(options, chronos2.WithBatchSize(int(f.next()%6)-1))
	}
	if flags&4 != 0 {
		options = append(options, chronos2.WithMaxParallelism(int(f.next()%5)-1))
	}
	if flags&8 != 0 {
		options = append(options, chronos2.WithLongHorizon())
	}
	if flags&16 != 0 {
		options = append(options, chronos2.WithCrossLearning())
	}
	return options
}

// checkForecast fails the test unless forecast has the model's native lines,
// each horizon steps long and finite.
func checkForecast(t *testing.T, label string, forecast chronos2.Forecast, levels, horizon int) {
	t.Helper()
	percents := forecast.NativePercentiles()
	if len(percents) != levels || forecast.Horizon() != horizon {
		t.Fatalf("%s: %d lines of %d steps, want %d of %d", label, len(percents), forecast.Horizon(), levels, horizon)
	}
	for _, percent := range percents {
		line, err := forecast.Percentile(percent)
		if err != nil {
			t.Fatalf("%s: Percentile(%v) error = %v", label, percent, err)
		}
		for step, value := range line {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				t.Fatalf("%s: p%v step %d = %v, want a finite value", label, percent, step, value)
			}
		}
	}
}

// FuzzForecast checks that Model.Forecast never panics and returns only documented errors for arbitrary histories and horizons.
func FuzzForecast(f *testing.F) {
	f.Add([]byte{0, 100, 0, 24, 0})
	f.Add([]byte{0, 3, 0, 1, 0xFF, 0xFF, 0xFF})
	f.Add([]byte{0, 40, 4, 1, 8, 0xFE})
	f.Add([]byte{0, 40, 4, 1, 8, 0xFC})
	f.Add([]byte{0, 40, 4, 1, 8, 1, 2, 0xFB, 3})
	f.Add([]byte{1, 200, 0, 30, 15, 0, 64, 3, 2})

	f.Fuzz(func(t *testing.T, data []byte) {
		// Given
		model := synthetic(t)
		input := &fuzzBytes{data: data}
		length := input.intn(fuzzMaxHistory)
		horizon := input.intn(fuzzMaxHorizon+2) - 1
		options := input.options()
		history := input.series(length)

		// When
		forecast, err := model.Forecast(t.Context(), history, horizon, options...)
		// Then
		if err != nil {
			if !matchesSentinel(err) {
				t.Fatalf("Forecast() error = %v, matches no sentinel", err)
			}
			return
		}
		checkForecast(t, "forecast", forecast, len(model.NativePercentiles()), horizon)
	})
}

// FuzzForecastTasks checks that Model.ForecastTasks never panics and returns only documented errors for arbitrary tasks.
func FuzzForecastTasks(f *testing.F) {
	f.Add([]byte{0, 50, 0, 12, 0, 2, 1, 1})
	f.Add([]byte{0, 30, 0, 8, 16, 3, 2, 2, 0, 0, 0})
	f.Add([]byte{0, 70, 0, 20, 1, 0, 40, 1, 3, 3, 1, 1, 7})
	f.Add(covariateSeed())

	f.Fuzz(func(t *testing.T, data []byte) {
		// Given
		model := synthetic(t)
		input := &fuzzBytes{data: data}
		length := input.intn(fuzzMaxHistory)
		horizon := input.intn(fuzzMaxHorizon+2) - 1
		options := input.options()
		tasks := make([]chronos2.Task, input.intn(3)+1)
		for i := range tasks {
			tasks[i] = fuzzTask(input, length, horizon)
		}

		// When
		results, err := model.ForecastTasks(t.Context(), tasks, horizon, options...)
		// Then
		if err != nil {
			if !matchesSentinel(err) {
				t.Fatalf("ForecastTasks() error = %v, matches no sentinel", err)
			}
			return
		}
		if len(results) != len(tasks) {
			t.Fatalf("%d results for %d tasks", len(results), len(tasks))
		}
		for i, forecasts := range results {
			if len(forecasts) != len(tasks[i].Targets) {
				t.Fatalf("task %d: %d forecasts for %d targets", i, len(forecasts), len(tasks[i].Targets))
			}
			for j, forecast := range forecasts {
				checkForecast(t, fmt.Sprintf("task %d target %d", i, j), forecast, len(model.NativePercentiles()), horizon)
			}
		}
	})
}

// fuzzTask decodes one task of up to fuzzMaxTargets targets with numeric and
// categorical covariates, mostly well formed but sometimes with mismatched
// lengths, unknown future keys, or names in both maps.
func fuzzTask(input *fuzzBytes, length, horizon int) chronos2.Task {
	targets := int(input.next())%fuzzMaxTargets + 1
	if input.next() == noTargetsByte {
		targets = 0
	}
	task := chronos2.Task{Targets: make([][]float64, targets)}
	for i := range task.Targets {
		task.Targets[i] = input.series(input.lengthNear(length))
	}
	flags := input.next()
	if flags&1 != 0 {
		task.Past = map[string][]float64{"p": input.series(input.lengthNear(length))}
	}
	if flags&2 != 0 {
		task.Future = map[string][]float64{"p": input.series(input.lengthNear(max(horizon, 0)))}
	}
	if flags&4 != 0 {
		task.Future = map[string][]float64{"unknown": input.series(max(horizon, 0))}
	}
	if flags&8 != 0 {
		task.PastCategorical = map[string][]string{"c": fuzzCategories(input, input.lengthNear(length))}
	}
	if flags&16 != 0 {
		task.FutureCategorical = map[string][]string{"c": fuzzCategories(input, input.lengthNear(max(horizon, 0)))}
	}
	if flags&32 != 0 {
		task.PastCategorical = map[string][]string{"p": fuzzCategories(input, length)}
	}
	return task
}

// fuzzCategories returns n category values.
func fuzzCategories(input *fuzzBytes, n int) []string {
	values := make([]string, n)
	for i := range values {
		values[i] = input.category()
	}
	return values
}

// covariateSeed encodes one well-formed two-target task of 20 steps with a
// known-future numeric covariate and a known-future categorical covariate,
// forecast 8 steps ahead, so the corpus starts from a successful call.
func covariateSeed() []byte {
	const length, horizon = 20, 8
	seed := []byte{0, length, 0, horizon + 1, 0, 0, 0}
	seed = append(seed, 1, 0) // two targets, not empty
	for target := range 2 {
		seed = append(seed, 1) // keep the expected length
		for i := range length {
			seed = append(seed, byte(0x60+target*length+i))
		}
	}
	seed = append(seed, 1|2|8|16) // past, future, past categorical, future categorical
	for _, steps := range []int{length, horizon, length, horizon} {
		seed = append(seed, 1) // keep the expected length
		for i := range steps {
			seed = append(seed, byte(0x70+i))
		}
	}
	return seed
}
