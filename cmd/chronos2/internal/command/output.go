package command

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Bounds of a requested percentile, inclusive.
const (
	// minPercentile is the lowest percentile the command accepts.
	minPercentile = 1.0

	// maxPercentile is the highest percentile the command accepts.
	maxPercentile = 99.0
)

// medianPercentile is the percentile shown as the median column.
const medianPercentile = 50.0

// parsePercentiles parses a comma-separated percentile list and returns the
// distinct values in ascending order with the median always included. It
// fails on an empty list or item, a non-number, or a value outside [1, 99].
func parsePercentiles(list string) ([]float64, error) {
	percentiles := []float64{medianPercentile}
	for item := range strings.SplitSeq(list, ",") {
		p, err := strconv.ParseFloat(strings.TrimSpace(item), 64)
		if err != nil || math.IsNaN(p) || p < minPercentile || p > maxPercentile {
			return nil, fmt.Errorf("%w: percentile %q is not a number in [1, 99]", ErrInvalidInput, item)
		}
		percentiles = append(percentiles, p)
	}
	slices.Sort(percentiles)
	return slices.Compact(percentiles), nil
}

// writeForecast writes result as CSV: a step column numbered from 1, then one
// column per percentile in the given ascending order, with 50 shown as median.
func writeForecast(w io.Writer, result Forecast, percentiles []float64) error {
	header := []string{"step"}
	lines := make([][]float64, len(percentiles))
	for i, p := range percentiles {
		if p == medianPercentile {
			header = append(header, "median")
			lines[i] = result.Median()
			continue
		}
		line, err := result.Percentile(p)
		if err != nil {
			return fmt.Errorf("percentile %v: %w", p, err)
		}
		header = append(header, "p"+strconv.FormatFloat(p, 'f', -1, 64))
		lines[i] = line
	}

	out := csv.NewWriter(w)
	if err := out.Write(header); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	for step := range result.Horizon() {
		record := []string{strconv.Itoa(step + 1)}
		for _, line := range lines {
			record = append(record, formatValue(line[step]))
		}
		if err := out.Write(record); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
	}
	out.Flush()
	if err := out.Error(); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

// formatValue renders a forecast value with the fewest digits that identify
// it at float32 precision, the precision the model computes in, so values
// read like 455.8 rather than 455.79998779296875.
func formatValue(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 32)
}
