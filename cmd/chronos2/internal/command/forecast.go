package command

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"

	"github.com/urfave/cli/v3"
)

// Flag names of the forecast subcommand.
const (
	// flagModel names the model to use.
	flagModel = "model"

	// flagHorizon names the number of steps to forecast.
	flagHorizon = "horizon"

	// flagPercentiles names the comma-separated percentile list.
	flagPercentiles = "percentiles"

	// flagContextLength names the cap on the history used.
	flagContextLength = "context-length"

	// flagLongHorizon names the switch permitting long horizons.
	flagLongHorizon = "long-horizon"

	// flagInput names the input file.
	flagInput = "input"
)

// defaultPercentiles is the --percentiles default.
const defaultPercentiles = "10,90"

// forecastCommand returns the forecast subcommand, which reads a column from
// stdin or --input and writes the forecast as CSV.
func forecastCommand(stdin io.Reader, forecast ForecastFunc) *cli.Command {
	return &cli.Command{
		Name:  "forecast",
		Usage: "forecast one numeric CSV column",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: flagModel, Value: ModelSmall, Usage: "model: small or base", Validator: validateModel},
			&cli.IntFlag{Name: flagHorizon, Required: true, Usage: "steps to forecast", Validator: requirePositive},
			&cli.StringFlag{Name: flagPercentiles, Value: defaultPercentiles, Usage: "comma-separated percentiles in [1, 99]"},
			&cli.IntFlag{Name: flagContextLength, Usage: "most recent observations to use (default: the model's)", Validator: requirePositive},
			&cli.BoolFlag{Name: flagLongHorizon, Usage: "permit horizons beyond one model pass"},
			&cli.StringFlag{Name: flagInput, Usage: "CSV file to read (default: standard input)"},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			return runForecast(ctx, cmd, stdin, forecast)
		},
	}
}

// runForecast parses the percentiles and input, forecasts, and writes the CSV
// to the command's writer. It rejects positional arguments, which the command
// does not take, so a file name passed without --input fails instead of
// leaving the command waiting on standard input.
func runForecast(ctx context.Context, cmd *cli.Command, stdin io.Reader, forecast ForecastFunc) error {
	if cmd.Args().Present() {
		return fmt.Errorf("%w: unexpected argument %q (pass a file with --%s)", ErrInvalidInput, cmd.Args().First(), flagInput)
	}
	percentiles, err := parsePercentiles(cmd.String(flagPercentiles))
	if err != nil {
		return err
	}
	history, err := readHistory(cmd.String(flagInput), stdin)
	if err != nil {
		return err
	}
	result, err := forecast(ctx, Request{
		Model:         cmd.String(flagModel),
		History:       history,
		Horizon:       cmd.Int(flagHorizon),
		ContextLength: cmd.Int(flagContextLength),
		LongHorizon:   cmd.Bool(flagLongHorizon),
	})
	if err != nil {
		return err
	}

	out := bufio.NewWriter(cmd.Root().Writer)
	if err := writeForecast(out, result, percentiles); err != nil {
		return err
	}
	if err := out.Flush(); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

// readHistory reads the series from the file at path, or from stdin when path
// is empty.
func readHistory(path string, stdin io.Reader) ([]float64, error) {
	if path == "" {
		return readSeries(stdin)
	}
	file, err := os.Open(path) // #nosec G304 -- the user names the file to read.
	if err != nil {
		return nil, fmt.Errorf("open input: %w", err)
	}
	history, readErr := readSeries(file)
	// A read-only file's close error carries nothing the read did not report.
	_ = file.Close()
	return history, readErr
}

// validateModel rejects a --model value other than ModelSmall or ModelBase.
func validateModel(name string) error {
	if name != ModelSmall && name != ModelBase {
		return fmt.Errorf("%w: model %q is not %q or %q", ErrInvalidInput, name, ModelSmall, ModelBase)
	}
	return nil
}

// requirePositive rejects a flag value below 1.
func requirePositive(value int) error {
	if value < 1 {
		return fmt.Errorf("%w: %d is not a positive number", ErrInvalidInput, value)
	}
	return nil
}
