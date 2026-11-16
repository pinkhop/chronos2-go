package command_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"strings"

	"github.com/pinkhop/chronos2-go/cmd/chronos2/internal/command"
)

// errUnknownPercentile is returned by fakeForecast for a level it holds no
// line for.
var errUnknownPercentile = errors.New("fake: unknown percentile")

// fakeForecast is a hand-written command.Forecast holding one line per
// percentile.
type fakeForecast struct {
	// lines maps each percentile, including 50, to its line.
	lines map[float64][]float64
}

// Horizon returns the length of the median line.
func (f fakeForecast) Horizon() int { return len(f.lines[50]) }

// Median returns the line at percentile 50.
func (f fakeForecast) Median() []float64 { return f.lines[50] }

// Percentile returns the line at p, or errUnknownPercentile when f holds none.
func (f fakeForecast) Percentile(p float64) ([]float64, error) {
	line, ok := f.lines[p]
	if !ok {
		return nil, fmt.Errorf("%w: %v", errUnknownPercentile, p)
	}
	return line, nil
}

// forecastSpy is a hand-written forecaster that records every request and
// returns a fixed result.
type forecastSpy struct {
	// result is returned for every request.
	result command.Forecast

	// failure, when non-nil, is returned instead of result.
	failure error

	// requests records each request in call order.
	requests []command.Request
}

// forecast records request and returns the spy's result or failure.
func (s *forecastSpy) forecast(_ context.Context, request command.Request) (command.Forecast, error) {
	s.requests = append(s.requests, request)
	if s.failure != nil {
		return nil, s.failure
	}
	return s.result, nil
}

// threeStepForecast returns a forecast of three steps with lines at 10, 25,
// 50, 90, and 97.5 whose values reveal their percentile.
func threeStepForecast() fakeForecast {
	return fakeForecast{lines: map[float64][]float64{
		10:   {1.5, 2.5, 3.5},
		25:   {4, 5, 6},
		50:   {7, 8, 9.25},
		90:   {10, 11, 12},
		97.5: {13, 14, 15},
	}}
}

// outcome captures one invocation of command.Run.
type outcome struct {
	// code is the exit status.
	code int

	// stdout holds everything written to standard output.
	stdout string

	// stderr holds everything written to standard error.
	stderr string
}

// invoke runs the command with args after the program name, feeding stdin and
// using spy and buildInfo as dependencies.
func invoke(spy *forecastSpy, stdin string, buildInfo func() (*debug.BuildInfo, bool), args ...string) outcome {
	var stdout, stderr bytes.Buffer
	dependencies := command.Dependencies{Forecast: spy.forecast, BuildInfo: buildInfo}
	arguments := append([]string{"chronos2"}, args...)
	code := command.Run(context.Background(), arguments, strings.NewReader(stdin), &stdout, &stderr, dependencies)
	return outcome{code: code, stdout: stdout.String(), stderr: stderr.String()}
}

// noBuildInfo is a build-information source that reports none available.
func noBuildInfo() (*debug.BuildInfo, bool) { return nil, false }

// failingWriter is an io.Writer whose every write fails.
type failingWriter struct{}

// errWriteFailed is the error every failingWriter write returns.
var errWriteFailed = errors.New("fake: write failed")

// Write always fails with errWriteFailed.
func (failingWriter) Write([]byte) (int, error) { return 0, errWriteFailed }
