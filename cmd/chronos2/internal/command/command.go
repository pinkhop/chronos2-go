// Package command implements the chronos2 command line: argument parsing,
// CSV input and output, and the version report. It reaches the models only
// through Dependencies, so the whole command is testable without loading one.
package command

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime/debug"

	"github.com/urfave/cli/v3"
)

// Model names accepted by the --model flag.
const (
	// ModelSmall selects the chronos-2-small checkpoint, the default.
	ModelSmall = "small"

	// ModelBase selects the chronos-2 base checkpoint.
	ModelBase = "base"
)

// exitFailure is the exit status for every failure.
const exitFailure = 1

// ErrInvalidInput reports a malformed input file or flag value.
var ErrInvalidInput = errors.New("invalid input")

// Forecast is the part of a chronos2.Forecast the command reads.
type Forecast interface {
	// Horizon returns the number of forecast steps.
	Horizon() int

	// Median returns the 50th-percentile line.
	Median() []float64

	// Percentile returns the line at percent p in [1, 99].
	Percentile(p float64) ([]float64, error)
}

// Request describes one univariate forecast.
type Request struct {
	// Model is ModelSmall or ModelBase.
	Model string

	// History holds the observed values, oldest first; NaN marks a missing
	// value.
	History []float64

	// Horizon is the number of steps to forecast, at least 1.
	Horizon int

	// ContextLength caps the history used; 0 means the model's default.
	ContextLength int

	// LongHorizon permits horizons beyond one model pass.
	LongHorizon bool
}

// ForecastFunc loads the requested model, when needed, and forecasts. Errors it
// returns are shown to the user verbatim.
type ForecastFunc func(ctx context.Context, request Request) (Forecast, error)

// Dependencies supplies the command's connections to the outside world.
type Dependencies struct {
	// Forecast runs a forecast. It must not be nil.
	Forecast ForecastFunc

	// BuildInfo reports the binary's build information, as
	// debug.ReadBuildInfo does. It must not be nil.
	BuildInfo func() (*debug.BuildInfo, bool)
}

// Run executes the command line args (program name first), reading input
// from stdin unless --input names a file and writing results to stdout and
// diagnostics to stderr. It returns the process exit status: 0 on success and
// non-zero on any failure, which it reports on stderr.
func Run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, dependencies Dependencies) int {
	root := &cli.Command{
		Name:  "chronos2",
		Usage: "forecast a CSV column with Chronos-2",
		Commands: []*cli.Command{
			forecastCommand(stdin, dependencies.Forecast),
			versionCommand(dependencies.BuildInfo),
		},
		Reader:    stdin,
		Writer:    stdout,
		ErrWriter: stderr,
		// Errors are reported below; the default handler would call os.Exit.
		ExitErrHandler: func(context.Context, *cli.Command, error) {},
	}
	if err := root.Run(ctx, args); err != nil {
		// A plain label rather than the program name: library errors already
		// begin with "chronos2:", which would otherwise appear twice. Nothing
		// more can be done if standard error itself fails.
		_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
		return exitFailure
	}
	return 0
}
