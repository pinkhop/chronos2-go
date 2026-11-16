// Command chronos2 forecasts a CSV column with Chronos-2 from the terminal.
// It embeds both the small and base models and loads only the one requested.
//
// Usage:
//
//	chronos2 forecast --horizon 24 [--model small|base] [--percentiles 10,90]
//	    [--context-length N] [--long-horizon] [--input file.csv]
//	chronos2 version
package main

import (
	"context"
	"os"
	"os/signal"
	"runtime/debug"

	"github.com/pinkhop/chronos2-go/cmd/chronos2/internal/command"
)

// main runs the command line, canceling the forecast on an interrupt, and
// exits with the command's status.
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	dependencies := command.Dependencies{Forecast: forecastWithModel, BuildInfo: debug.ReadBuildInfo}
	code := command.Run(ctx, os.Args, os.Stdin, os.Stdout, os.Stderr, dependencies)
	stop()
	os.Exit(code)
}
