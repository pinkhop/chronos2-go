package chronos2

import (
	"context"
	"math"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// testLevels are the small model's native quantile levels.
var testLevels = []float64{0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}

// Dimensions of the fake model: a 64-step context, 16-step patches, at most
// four output patches (a 64-step single-pass horizon), and width 4.
const (
	testContextLength = 64
	testPatchSize     = 16
	testMaxPatches    = 4
	testWidth         = 4
)

// testSpec describes the fake model.
var testSpec = modelSpec{
	name:               "fake",
	contextLength:      testContextLength,
	outputPatchSize:    testPatchSize,
	maxOutputPatches:   testMaxPatches,
	quantiles:          testLevels,
	embeddingDimension: testWidth,
	regToken:           true,
}

// fakeForwarder is a spy standing in for the network. Its predictions encode
// where they came from: row r, level q, step s predicts marker(r) + q +
// s/1000, where marker(r) is the row's last observed context value, so tests
// can check routing and truncation. Hidden states hold token×1000 + element.
type fakeForwarder struct {
	// batches records every batch passed to Forward.
	batches []network.Batch

	// options records the options of every call.
	options []forwardOptions

	// err, when set, is returned by Forward instead of a prediction.
	err error

	// extraRows, when nonzero, makes Forward report a wrong row count.
	extraRows int

	// afterForward, when set, runs at the end of every successful Forward,
	// letting a test act between engine passes. It may be nil.
	afterForward func()
}

// Forward records the call and returns the fake prediction.
func (f *fakeForwarder) Forward(_ context.Context, batch network.Batch, options forwardOptions) (network.Output, error) {
	f.batches = append(f.batches, batch)
	f.options = append(f.options, options)
	if f.err != nil {
		return network.Output{}, f.err
	}
	if f.afterForward != nil {
		defer f.afterForward()
	}
	steps := batch.OutputPatches * testPatchSize
	output := network.Output{
		Rows:      batch.Rows + f.extraRows,
		Quantiles: len(testLevels),
		Steps:     steps,
		Locations: make([]float32, batch.Rows),
		Scales:    make([]float32, batch.Rows),
	}
	for row := range batch.Rows {
		marker := lastObserved(batch.Context[row*batch.ContextLength : (row+1)*batch.ContextLength])
		output.Locations[row], output.Scales[row] = marker, 2
		for level := range len(testLevels) {
			for step := range steps {
				output.Values = append(output.Values, marker+float32(level)+float32(step)/1000)
			}
		}
	}
	if options.hidden {
		output.Tokens = (batch.ContextLength+testPatchSize-1)/testPatchSize + 1 + batch.OutputPatches
		output.Width = testWidth
		for token := range output.Tokens {
			for element := range testWidth {
				output.Hidden = append(output.Hidden, float32(token*1000+element))
			}
		}
	}
	return output, nil
}

// lastObserved returns the last non-NaN value of row, or NaN.
func lastObserved(row []float32) float32 {
	for _, value := range slices.Backward(row) {
		if !math.IsNaN(float64(value)) {
			return value
		}
	}
	return float32(math.NaN())
}

// unrollSpy records long-horizon calls and returns marker-encoded values like
// fakeForwarder, with steps continuing past one pass.
type unrollSpy struct {
	// horizons records the horizon of every call.
	horizons []int
}

// unroll is a longHorizonForecaster.
func (u *unrollSpy) unroll(_ context.Context, _ forwarder, batch batching.Batch, horizon, _ int) ([]float32, error) {
	u.horizons = append(u.horizons, horizon)
	values := make([]float32, 0, batch.Rows*len(testLevels)*horizon)
	for row := range batch.Rows {
		marker := lastObserved(batch.Context[row*batch.ContextLength : (row+1)*batch.ContextLength])
		for level := range len(testLevels) {
			for step := range horizon {
				values = append(values, marker+float32(level)+float32(step)/1000)
			}
		}
	}
	return values, nil
}

// newFakeModel returns a Model over a fresh fakeForwarder and unrollSpy.
func newFakeModel() (*Model, *fakeForwarder, *unrollSpy) {
	forward, spy := &fakeForwarder{}, &unrollSpy{}
	return newModel(testSpec, forward, spy.unroll), forward, spy
}

// expectedLine returns the fake prediction line for marker at level index
// level over horizon steps.
func expectedLine(marker float32, level, horizon int) []float64 {
	line := make([]float64, horizon)
	for step := range line {
		line[step] = float64(marker + float32(level) + float32(step)/1000)
	}
	return line
}

// ramp returns n values start, start+1, ….
func ramp(start float64, n int) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = start + float64(i)
	}
	return values
}
