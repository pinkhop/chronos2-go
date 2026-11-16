package network

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/testmodel"
)

// instanceNormCase is one reference vector: a row as fed to the model, with
// null for NaN, and the reference's float32 loc and scale.
type instanceNormCase struct {
	// Row holds the row's values; nil entries are NaN.
	Row []*float64 `json:"row"`

	// Loc is the reference location.
	Loc float64 `json:"loc"`

	// Scale is the reference scale.
	Scale float64 `json:"scale"`
}

// readInstanceNormCases reads testdata/instance_norm.json, failing the test
// when it cannot.
func readInstanceNormCases(t *testing.T) []instanceNormCase {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "instance_norm.json"))
	if err != nil {
		t.Fatalf("read vectors: %v", err)
	}
	var vectors struct {
		Cases []instanceNormCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatalf("decode vectors: %v", err)
	}
	return vectors.Cases
}

// float32Row converts a case's row to float32 with NaN for nil.
func float32Row(row []*float64) []float32 {
	values := make([]float32, len(row))
	for i, value := range row {
		if value == nil {
			values[i] = float32(math.NaN())
			continue
		}
		values[i] = float32(*value)
	}
	return values
}

// constantValue reports whether every observed value of row is the same and,
// if so, returns that value. A row with no observations is not constant.
func constantValue(row []float32) (float32, bool) {
	observed := slices.DeleteFunc(slices.Clone(row), isNaN)
	if len(observed) == 0 {
		return 0, false
	}
	for _, value := range observed[1:] {
		if value != observed[0] {
			return 0, false
		}
	}
	return observed[0], true
}

// rowStatisticsResult is one row's location and scale from rowStatistics.
type rowStatisticsResult struct {
	// location is the row's computed location.
	location float32

	// scale is the row's computed scale.
	scale float32
}

// TestRowStatistics_NonConstantReferenceRows_MatchTorchWithinStageTolerance
// verifies that rowStatistics agrees with PyTorch's InstanceNorm, within the
// stage tolerance, on every reference row that is not constant.
func TestRowStatistics_NonConstantReferenceRows_MatchTorchWithinStageTolerance(t *testing.T) {
	t.Parallel()

	// Given the reference rows with more than one distinct observed value
	var rows [][]float32
	var wants []instanceNormCase
	for _, vector := range readInstanceNormCases(t) {
		row := float32Row(vector.Row)
		if _, constant := constantValue(row); !constant {
			rows, wants = append(rows, row), append(wants, vector)
		}
	}
	if len(rows) == 0 {
		t.Fatalf("no non-constant reference rows")
	}

	// When
	results := make([]rowStatisticsResult, len(rows))
	for i, row := range rows {
		results[i].location, results[i].scale = rowStatistics(row)
	}

	// Then
	for i, got := range results {
		want := wants[i]
		if !testmodel.Within(float64(got.location), want.Loc, testmodel.StageAbsTolerance, testmodel.StageRelTolerance) ||
			!testmodel.Within(float64(got.scale), want.Scale, testmodel.StageAbsTolerance, testmodel.StageRelTolerance) {
			t.Errorf("row %d (%d values): loc, scale = %v, %v; want %v, %v",
				i, len(rows[i]), got.location, got.scale, want.Loc, want.Scale)
		}
	}
}

// TestRowStatistics_ConstantReferenceRows_ReturnExactMeanAndMinimumScale
// verifies that rowStatistics gives every constant reference row its exact
// value as location and minimumScale as scale. This is a documented deviation:
// the reference's float32 mean can land one ulp off, giving a tiny nonzero
// scale instead.
func TestRowStatistics_ConstantReferenceRows_ReturnExactMeanAndMinimumScale(t *testing.T) {
	t.Parallel()

	// Given the reference rows whose observed values are all equal
	var rows [][]float32
	var values []float32
	for _, vector := range readInstanceNormCases(t) {
		row := float32Row(vector.Row)
		if value, constant := constantValue(row); constant {
			rows, values = append(rows, row), append(values, value)
		}
	}
	if len(rows) == 0 {
		t.Fatalf("no constant reference rows")
	}

	// When
	results := make([]rowStatisticsResult, len(rows))
	for i, row := range rows {
		results[i].location, results[i].scale = rowStatistics(row)
	}

	// Then
	for i, got := range results {
		if got.location != values[i] || got.scale != minimumScale {
			t.Errorf("row %d (%d values): loc, scale = %v, %v; want %v, %v",
				i, len(rows[i]), got.location, got.scale, values[i], float32(minimumScale))
		}
	}
}

// TestRowStatistics_AllNaN_ReturnsZeroLocationAndUnitScale verifies that rowStatistics gives an all-NaN row location 0 and scale 1.
func TestRowStatistics_AllNaN_ReturnsZeroLocationAndUnitScale(t *testing.T) {
	t.Parallel()

	// Given
	nan := float32(math.NaN())
	row := []float32{nan, nan, nan, nan, nan}

	// When
	location, scale := rowStatistics(row)

	// Then
	if location != 0 || scale != 1 {
		t.Errorf("rowStatistics() = %v, %v; want 0, 1", location, scale)
	}
}

// TestRowStatistics_ExactConstantRow_ReturnsMinimumScale verifies that rowStatistics gives an exactly constant row the minimum scale.
func TestRowStatistics_ExactConstantRow_ReturnsMinimumScale(t *testing.T) {
	t.Parallel()

	// Given a constant row whose float32 mean is exact.
	row := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2}

	// When
	location, scale := rowStatistics(row)

	// Then
	if location != 2 || scale != minimumScale {
		t.Errorf("rowStatistics() = %v, %v; want 2, %v", location, scale, float32(minimumScale))
	}
}
