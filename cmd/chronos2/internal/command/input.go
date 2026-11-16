package command

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// byteOrderMark is the UTF-8 byte-order mark some editors put at the start of
// a CSV file.
const byteOrderMark = "\xef\xbb\xbf"

// readSeries reads one numeric CSV column. A non-numeric first row is a header
// and is skipped; a blank cell or NaN (in any case) marks a missing value.
// Trailing blank lines are ignored, so a file's final newlines add no
// missing values. It fails on a row holding more than one column or, after the
// first row, a non-numeric cell.
func readSeries(r io.Reader) ([]float64, error) {
	scanner := bufio.NewScanner(r)
	var values []float64
	observedLength := 0 // length of values through the last non-blank cell
	for line := 1; scanner.Scan(); line++ {
		text := scanner.Text()
		if line == 1 {
			text = strings.TrimPrefix(text, byteOrderMark)
		}
		cell, err := parseCell(text, line)
		if err != nil {
			return nil, err
		}
		switch {
		case cell.header:
			continue
		case cell.blank:
			values = append(values, math.NaN())
		default:
			values = append(values, cell.value)
			observedLength = len(values)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read input: %w", err)
	}
	return values[:observedLength], nil
}

// cell is one parsed row of the input column.
type cell struct {
	// value is the parsed number, NaN for an explicit NaN.
	value float64

	// blank reports an empty cell, which marks a missing value.
	blank bool

	// header reports a non-numeric first row.
	header bool
}

// parseCell parses the row text found on 1-based line.
func parseCell(text string, line int) (cell, error) {
	trimmed := strings.TrimSpace(text)
	if field, ok := unquoteField(trimmed); ok {
		trimmed = strings.TrimSpace(field)
	} else if strings.Contains(trimmed, ",") {
		return cell{}, fmt.Errorf("%w: line %d: expected one column", ErrInvalidInput, line)
	}
	if trimmed == "" {
		return cell{blank: true}, nil
	}
	if strings.EqualFold(trimmed, "nan") {
		return cell{value: math.NaN()}, nil
	}
	value, err := strconv.ParseFloat(trimmed, 64)
	switch {
	case err == nil:
		return cell{value: value}, nil
	case line == 1:
		return cell{header: true}, nil
	default:
		return cell{}, fmt.Errorf("%w: line %d: %q is not a number", ErrInvalidInput, line, trimmed)
	}
}

// csvQuote is the CSV quote character; a doubled quote inside a quoted field
// stands for one literal quote.
const csvQuote = `"`

// unquoteField returns the content of text, with doubled quotes unescaped,
// when text is exactly one quoted CSV field. It reports false for unquoted
// text and for several quoted fields such as "a","b", whose inner quotes are
// not doubled.
func unquoteField(text string) (string, bool) {
	if len(text) < 2 || !strings.HasPrefix(text, csvQuote) || !strings.HasSuffix(text, csvQuote) {
		return "", false
	}
	inner := text[1 : len(text)-1]
	if strings.Contains(strings.ReplaceAll(inner, csvQuote+csvQuote, ""), csvQuote) {
		return "", false
	}
	return strings.ReplaceAll(inner, csvQuote+csvQuote, csvQuote), true
}
