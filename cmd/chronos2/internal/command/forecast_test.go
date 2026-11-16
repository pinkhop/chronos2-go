package command_test

import (
	"bytes"
	"context"
	"errors"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/pinkhop/chronos2-go/cmd/chronos2/internal/command"
)

// TestRun_DefaultPercentiles_WritesStepP10MedianP90 verifies that Run writes step, p10, median, and p90 columns by default.
func TestRun_DefaultPercentiles_WritesStepP10MedianP90(t *testing.T) {
	t.Parallel()

	// Given a three-step forecast
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast with default flags
	result := invoke(spy, "1\n2\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the CSV holds steps from 1 with p10, median, and p90 columns
	want := "step,p10,median,p90\n1,1.5,7,10\n2,2.5,8,11\n3,3.5,9.25,12\n"
	if result.code != 0 || result.stdout != want {
		t.Errorf("code = %d, stdout = %q, want %q (stderr %q)", result.code, result.stdout, want, result.stderr)
	}
}

// TestRun_UnsortedPercentilesWithFiftyAndDuplicates_OrdersAscendingWithMedianOnce verifies that Run, given unsorted percentiles with fifty and duplicates, orders ascending with median once.
func TestRun_UnsortedPercentilesWithFiftyAndDuplicates_OrdersAscendingWithMedianOnce(t *testing.T) {
	t.Parallel()

	// Given requested percentiles out of order, with 50 and a duplicate
	spy := &forecastSpy{result: threeStepForecast()}

	// When they are forecast
	result := invoke(spy, "1\n", noBuildInfo, "forecast", "--horizon", "3", "--percentiles", "97.5, 25,50,25")

	// Then columns ascend, 50 appears only as median, and duplicates collapse
	header, _, _ := strings.Cut(result.stdout, "\n")
	if result.code != 0 || header != "step,p25,median,p97.5" {
		t.Errorf("code = %d, header = %q, want %q (stderr %q)", result.code, header, "step,p25,median,p97.5", result.stderr)
	}
}

// TestRun_PercentilesOutOfRange_FailsWithoutForecasting verifies that Run, given percentiles out of range, fails without forecasting.
func TestRun_PercentilesOutOfRange_FailsWithoutForecasting(t *testing.T) {
	t.Parallel()

	for _, percentiles := range []string{"0.5", "99.5", "10,", "abc", "NaN", ""} {
		t.Run(percentiles, func(t *testing.T) {
			t.Parallel()

			// Given an invalid percentile list
			spy := &forecastSpy{result: threeStepForecast()}

			// When it is forecast
			result := invoke(spy, "1\n", noBuildInfo, "forecast", "--horizon", "3", "--percentiles", percentiles)

			// Then the command fails before forecasting
			if result.code == 0 || result.stderr == "" || len(spy.requests) != 0 {
				t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
			}
		})
	}
}

// TestRun_AllFlags_PassedToForecaster verifies that Run passes every forecasting flag to the forecaster.
func TestRun_AllFlags_PassedToForecaster(t *testing.T) {
	t.Parallel()

	// Given every forecasting flag
	spy := &forecastSpy{result: threeStepForecast()}

	// When the command runs
	result := invoke(spy, "1\n2\n", noBuildInfo, "forecast",
		"--model", "base", "--horizon", "3", "--context-length", "512", "--long-horizon")

	// Then the forecaster receives each value
	if result.code != 0 || len(spy.requests) != 1 {
		t.Fatalf("code = %d, stderr = %q", result.code, result.stderr)
	}
	got := spy.requests[0]
	if got.Model != command.ModelBase || got.Horizon != 3 || got.ContextLength != 512 || !got.LongHorizon {
		t.Errorf("request = %+v", got)
	}
}

// TestRun_DefaultFlags_RequestSmallModelWithModelDefaults verifies that Run, given only a horizon, requests the small model with its default settings.
func TestRun_DefaultFlags_RequestSmallModelWithModelDefaults(t *testing.T) {
	t.Parallel()

	// Given only the required horizon
	spy := &forecastSpy{result: threeStepForecast()}

	// When the command runs
	result := invoke(spy, "1\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the small model, its own context length, and no long horizon apply
	if result.code != 0 || len(spy.requests) != 1 {
		t.Fatalf("code = %d, stderr = %q", result.code, result.stderr)
	}
	got := spy.requests[0]
	if got.Model != command.ModelSmall || got.ContextLength != 0 || got.LongHorizon {
		t.Errorf("request = %+v", got)
	}
}

// TestRun_InvalidFlags_FailWithoutForecasting verifies that Run, given invalid flags, fails without forecasting.
func TestRun_InvalidFlags_FailWithoutForecasting(t *testing.T) {
	t.Parallel()

	cases := map[string][]string{
		"missing horizon": {"forecast"},
		"zero horizon":    {"forecast", "--horizon", "0"},
		"unknown model":   {"forecast", "--horizon", "3", "--model", "large"},
		"zero context":    {"forecast", "--horizon", "3", "--context-length", "0"},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Given invalid flags
			spy := &forecastSpy{result: threeStepForecast()}

			// When the command runs
			result := invoke(spy, "1\n", noBuildInfo, args...)

			// Then it fails before forecasting
			if result.code == 0 || result.stderr == "" || len(spy.requests) != 0 {
				t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
			}
		})
	}
}

// TestRun_ForecastWithPositionalArgument_FailsPointingAtInputFlag verifies
// that Run rejects a stray positional argument, such as a file name passed
// without --input, instead of ignoring it and reading standard input.
func TestRun_ForecastWithPositionalArgument_FailsPointingAtInputFlag(t *testing.T) {
	t.Parallel()

	// Given a file name passed as a positional argument
	spy := &forecastSpy{result: threeStepForecast()}

	// When the forecast runs
	result := invoke(spy, "1\n2\n", noBuildInfo, "forecast", "--horizon", "3", "data.csv")

	// Then the command fails naming the argument and --input, without forecasting
	if result.code == 0 || !strings.Contains(result.stderr, `"data.csv"`) || !strings.Contains(result.stderr, "--input") ||
		len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}

// TestRun_ForecasterFails_ReportsErrorOnStderr verifies that Run reports a forecaster failure on standard error.
func TestRun_ForecasterFails_ReportsErrorOnStderr(t *testing.T) {
	t.Parallel()

	// Given a forecaster that fails
	spy := &forecastSpy{failure: errors.New("horizon too long")}

	// When the command runs
	result := invoke(spy, "1\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the error goes to stderr with a non-zero exit and nothing to stdout
	if result.code == 0 || !strings.Contains(result.stderr, "horizon too long") || result.stdout != "" {
		t.Errorf("code = %d, stdout = %q, stderr = %q", result.code, result.stdout, result.stderr)
	}
}

// TestRun_ForecasterFails_PrintsErrorOnceWithoutProgramPrefix verifies that
// Run prints a forecaster error after a plain "error: " label, so a library
// message that already names chronos2 is not prefixed twice.
func TestRun_ForecasterFails_PrintsErrorOnceWithoutProgramPrefix(t *testing.T) {
	t.Parallel()

	// Given a forecaster failing with a message that already names the library
	spy := &forecastSpy{failure: errors.New("chronos2: horizon exceeds the model's single-pass maximum")}

	// When the command runs
	result := invoke(spy, "1\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then standard error holds exactly one labeled line
	want := "error: chronos2: horizon exceeds the model's single-pass maximum\n"
	if result.code == 0 || result.stderr != want {
		t.Errorf("code = %d, stderr = %q, want %q", result.code, result.stderr, want)
	}
}

// TestRun_StdoutWriteFails_ExitsNonZero verifies that Run exits non-zero when standard output rejects writes.
func TestRun_StdoutWriteFails_ExitsNonZero(t *testing.T) {
	t.Parallel()

	// Given standard output that rejects every write
	spy := &forecastSpy{result: threeStepForecast()}
	var stderr bytes.Buffer
	dependencies := command.Dependencies{Forecast: spy.forecast, BuildInfo: noBuildInfo}
	arguments := []string{"chronos2", "forecast", "--horizon", "3"}

	// When the command runs
	code := command.Run(context.Background(), arguments, strings.NewReader("1\n"), failingWriter{}, &stderr, dependencies)

	// Then the failure is reported
	if code == 0 || stderr.Len() == 0 {
		t.Errorf("code = %d, stderr = %q", code, stderr.String())
	}
}

// TestRun_Version_PrintsModuleVersionAndRevision verifies that the version command prints the module version and VCS revision.
func TestRun_Version_PrintsModuleVersionAndRevision(t *testing.T) {
	t.Parallel()

	// Given build information with a version and a modified revision
	buildInfo := func() (*debug.BuildInfo, bool) {
		return &debug.BuildInfo{
			Main: debug.Module{Version: "v1.2.3"},
			Settings: []debug.BuildSetting{
				{Key: "vcs.revision", Value: "abc123"},
				{Key: "vcs.modified", Value: "true"},
			},
		}, true
	}

	// When the version command runs
	result := invoke(&forecastSpy{}, "", buildInfo, "version")

	// Then it prints the version, the revision, and the modified marker
	want := "chronos2 v1.2.3\nrevision abc123 (modified)\n"
	if result.code != 0 || result.stdout != want {
		t.Errorf("code = %d, stdout = %q, want %q", result.code, result.stdout, want)
	}
}

// TestRun_VersionWithoutBuildInfo_PrintsUnknown verifies that the version command prints an unknown version without build information.
func TestRun_VersionWithoutBuildInfo_PrintsUnknown(t *testing.T) {
	t.Parallel()

	// Given no build information
	buildInfo := noBuildInfo

	// When the version command runs
	result := invoke(&forecastSpy{}, "", buildInfo, "version")

	// Then it reports an unknown version and no revision
	if want := "chronos2 unknown\n"; result.code != 0 || result.stdout != want {
		t.Errorf("code = %d, stdout = %q, want %q", result.code, result.stdout, want)
	}
}
