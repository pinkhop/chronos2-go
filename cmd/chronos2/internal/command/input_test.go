package command_test

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// sameValues reports whether a and b hold the same values in order, treating
// any two NaN values as equal.
func sameValues(a, b []float64) bool {
	return slices.EqualFunc(a, b, func(x, y float64) bool {
		return x == y || (math.IsNaN(x) && math.IsNaN(y))
	})
}

// historyFor runs a forecast over stdin and returns the history the
// forecaster received, failing t when the command does not succeed.
func historyFor(t *testing.T, stdin string) []float64 {
	t.Helper()
	spy := &forecastSpy{result: threeStepForecast()}
	result := invoke(spy, stdin, noBuildInfo, "forecast", "--horizon", "3")
	if result.code != 0 || len(spy.requests) != 1 {
		t.Fatalf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
	return spy.requests[0].History
}

// TestRun_InputWithHeader_SkipsHeaderRow verifies that Run, given input with header, skips header row.
func TestRun_InputWithHeader_SkipsHeaderRow(t *testing.T) {
	t.Parallel()

	// Given a column with a non-numeric header
	stdin := "load\n1\n2.5\n-3e2\n"

	// When it is forecast
	history := historyFor(t, stdin)

	// Then the header is skipped and the values parsed
	if want := []float64{1, 2.5, -300}; !sameValues(history, want) {
		t.Errorf("history = %v, want %v", history, want)
	}
}

// TestRun_InputWithBlankAndNaNCells_MarksThemMissing verifies that Run, given input with blank and NaN cells, marks them missing.
func TestRun_InputWithBlankAndNaNCells_MarksThemMissing(t *testing.T) {
	t.Parallel()

	// Given a header-less column with a blank cell and NaN in two spellings
	stdin := "NaN\n1\n\n2\nnan\n3\n"

	// When it is forecast
	history := historyFor(t, stdin)

	// Then each blank or NaN cell becomes a missing value, and a leading NaN
	// is not mistaken for a header
	want := []float64{math.NaN(), 1, math.NaN(), 2, math.NaN(), 3}
	if !sameValues(history, want) {
		t.Errorf("history = %v, want %v", history, want)
	}
}

// TestRun_InputWithTrailingBlankLines_IgnoresThem verifies that Run, given input with trailing blank lines, ignores them.
func TestRun_InputWithTrailingBlankLines_IgnoresThem(t *testing.T) {
	t.Parallel()

	// Given a column followed by blank lines
	stdin := "1\n2\n\n  \n"

	// When it is forecast
	history := historyFor(t, stdin)

	// Then the trailing blank lines add no values
	if want := []float64{1, 2}; !sameValues(history, want) {
		t.Errorf("history = %v, want %v", history, want)
	}
}

// TestRun_InputWithWindowsLineEndingsQuotesAndBOM_ParsesValues verifies that Run parses input with Windows line endings, quotes, and a byte-order mark.
func TestRun_InputWithWindowsLineEndingsQuotesAndBOM_ParsesValues(t *testing.T) {
	t.Parallel()

	// Given a byte-order mark, CRLF line endings, a quoted header, and padding
	stdin := "\xef\xbb\xbf\"load\"\r\n\"1\"\r\n 2 \r\n"

	// When it is forecast
	history := historyFor(t, stdin)

	// Then the values parse cleanly
	if want := []float64{1, 2}; !sameValues(history, want) {
		t.Errorf("history = %v, want %v", history, want)
	}
}

// TestRun_InputWithNonNumericRowAfterFirst_FailsNamingLine verifies that Run, given a non-numeric row after the first, fails naming the line.
func TestRun_InputWithNonNumericRowAfterFirst_FailsNamingLine(t *testing.T) {
	t.Parallel()

	// Given a non-numeric cell on line 3
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast
	result := invoke(spy, "load\n1\nabc\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the command fails, names the line, and never forecasts
	if result.code == 0 || !strings.Contains(result.stderr, "line 3") || len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}

// TestRun_InputWithSeveralColumns_Fails verifies that Run, given input with several columns, fails.
func TestRun_InputWithSeveralColumns_Fails(t *testing.T) {
	t.Parallel()

	// Given a two-column CSV
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast
	result := invoke(spy, "1,2\n3,4\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the command fails without forecasting
	if result.code == 0 || !strings.Contains(result.stderr, "one column") || len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}

// TestRun_InputWithSeveralQuotedHeaderColumns_Fails verifies that Run rejects
// a header row of several quoted columns instead of skipping it as one header.
func TestRun_InputWithSeveralQuotedHeaderColumns_Fails(t *testing.T) {
	t.Parallel()

	// Given a CSV whose header row holds two quoted columns
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast
	result := invoke(spy, "\"date\",\"load\"\n1\n2\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the command fails naming the column count, without forecasting
	if result.code == 0 || !strings.Contains(result.stderr, "line 1: expected one column") || len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}

// TestRun_InputWithSeveralQuotedDataColumns_FailsNamingColumnCount verifies
// that Run reports a data row of several quoted columns as a column-count
// error rather than as a non-numeric cell.
func TestRun_InputWithSeveralQuotedDataColumns_FailsNamingColumnCount(t *testing.T) {
	t.Parallel()

	// Given a data row on line 2 holding two quoted columns
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast
	result := invoke(spy, "load\n\"1\",\"2\"\n", noBuildInfo, "forecast", "--horizon", "3")

	// Then the command fails naming the column count, without forecasting
	if result.code == 0 || !strings.Contains(result.stderr, "line 2: expected one column") || len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}

// TestRun_InputWithQuotedCommaAndEscapedQuote_ParsesAsOneColumn verifies that
// Run treats a quoted cell holding a comma or a doubled quote as one column.
func TestRun_InputWithQuotedCommaAndEscapedQuote_ParsesAsOneColumn(t *testing.T) {
	t.Parallel()

	// Given a quoted header holding a comma and an escaped quote, then values
	stdin := "\"load, \"\"kW\"\"\"\n\"1\"\n2\n"

	// When it is forecast
	history := historyFor(t, stdin)

	// Then the header is skipped as one column and the values parsed
	if want := []float64{1, 2}; !sameValues(history, want) {
		t.Errorf("history = %v, want %v", history, want)
	}
}

// TestRun_InputFlag_ReadsNamedFileInsteadOfStdin verifies that Run reads the --input file instead of standard input.
func TestRun_InputFlag_ReadsNamedFileInsteadOfStdin(t *testing.T) {
	t.Parallel()

	// Given a file holding a column and unrelated standard input
	path := filepath.Join(t.TempDir(), "load.csv")
	if err := os.WriteFile(path, []byte("value\n4\n5\n"), 0o600); err != nil {
		t.Fatalf("write input: %v", err)
	}
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast with --input
	result := invoke(spy, "99\n", noBuildInfo, "forecast", "--horizon", "3", "--input", path)

	// Then the file's values are forecast
	if result.code != 0 || len(spy.requests) != 1 {
		t.Fatalf("code = %d, stderr = %q", result.code, result.stderr)
	}
	if want := []float64{4, 5}; !sameValues(spy.requests[0].History, want) {
		t.Errorf("history = %v, want %v", spy.requests[0].History, want)
	}
}

// TestRun_InputFileMissing_Fails verifies that Run fails when the --input file does not exist.
func TestRun_InputFileMissing_Fails(t *testing.T) {
	t.Parallel()

	// Given a path to no file
	path := filepath.Join(t.TempDir(), "absent.csv")
	spy := &forecastSpy{result: threeStepForecast()}

	// When it is forecast with --input
	result := invoke(spy, "", noBuildInfo, "forecast", "--horizon", "3", "--input", path)

	// Then the command fails without forecasting
	if result.code == 0 || result.stderr == "" || len(spy.requests) != 0 {
		t.Errorf("code = %d, stderr = %q, requests = %d", result.code, result.stderr, len(spy.requests))
	}
}
