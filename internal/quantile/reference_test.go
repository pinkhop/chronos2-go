package quantile

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// referenceVectors mirrors testdata/reference.json, which holds inputs and
// outputs computed by the chronos-forecasting 2.3.2 reference in float32.
type referenceVectors struct {
	// Interpolate holds interpolate_quantiles cases.
	Interpolate []struct {
		Levels          []float64   `json:"levels"`
		Queries         []float64   `json:"queries"`
		ValuesByLevel   [][]float32 `json:"values_by_level"`
		ExpectedByQuery [][]float32 `json:"expected_by_query"`
	} `json:"interpolate"`
	// WeightedQuantile holds weighted_quantile cases, one step each.
	WeightedQuantile []struct {
		Queries  []float64 `json:"queries"`
		Weights  []float64 `json:"weights"`
		Samples  []float32 `json:"samples"`
		Expected []float32 `json:"expected"`
	} `json:"weighted_quantile"`
	// LevelMass holds _get_prob_mass_per_quantile_level cases.
	LevelMass []struct {
		Levels []float64 `json:"levels"`
		Mass   []float64 `json:"mass"`
	} `json:"level_mass"`
}

// referenceTolerance bounds the difference from the reference; the
// reference computes in float32, so a few units in the last place remain.
const referenceTolerance = 1e-5

// readReferenceVectors loads testdata/reference.json or fails the test.
func readReferenceVectors(t *testing.T) referenceVectors {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "reference.json"))
	if err != nil {
		t.Fatalf("read reference vectors: %v", err)
	}
	var vectors referenceVectors
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatalf("decode reference vectors: %v", err)
	}
	return vectors
}

// closeTo reports whether got is within referenceTolerance of want, relative
// to want's magnitude when it exceeds 1.
func closeTo(got, want float64) bool {
	return math.Abs(got-want) <= referenceTolerance*math.Max(1, math.Abs(want))
}

// TestInterpolate_ReferenceVectors_MatchReference verifies that Interpolate matches the reference implementation's vectors.
func TestInterpolate_ReferenceVectors_MatchReference(t *testing.T) {
	t.Parallel()

	// Given: the reference interpolation cases.
	vectors := readReferenceVectors(t)

	for i, c := range vectors.Interpolate {
		// When: each case is interpolated.
		got := Interpolate(c.Queries, c.Levels, c.ValuesByLevel)

		// Then: every value matches the reference.
		for q := range c.ExpectedByQuery {
			for step, want := range c.ExpectedByQuery[q] {
				if !closeTo(float64(got[q][step]), float64(want)) {
					t.Errorf("case %d query %v step %d: got %v, want %v", i, c.Queries[q], step, got[q][step], want)
				}
			}
		}
	}
}

// TestWeightedQuantile_ReferenceVectors_MatchReference verifies that WeightedQuantile matches the reference implementation's vectors.
func TestWeightedQuantile_ReferenceVectors_MatchReference(t *testing.T) {
	t.Parallel()

	// Given: the reference weighted-quantile cases (nine unrolled deciles
	// times the native levels, with outer-product weights).
	vectors := readReferenceVectors(t)

	for i, c := range vectors.WeightedQuantile {
		// When: each case is reduced.
		got := WeightedQuantile(c.Queries, c.Weights, c.Samples)

		// Then: every level matches the reference.
		for q, want := range c.Expected {
			if !closeTo(float64(got[q]), float64(want)) {
				t.Errorf("case %d level %v: got %v, want %v", i, c.Queries[q], got[q], want)
			}
		}
	}
}

// TestLevelMass_ReferenceVectors_MatchReference verifies that LevelMass matches the reference implementation's vectors.
func TestLevelMass_ReferenceVectors_MatchReference(t *testing.T) {
	t.Parallel()

	// Given: the reference level-mass cases for both models' levels.
	vectors := readReferenceVectors(t)

	for i, c := range vectors.LevelMass {
		// When: each level set's mass is computed.
		got := LevelMass(c.Levels)

		// Then: every mass matches the reference.
		for j, want := range c.Mass {
			if !closeTo(got[j], want) {
				t.Errorf("case %d level %v: got %v, want %v", i, c.Levels[j], got[j], want)
			}
		}
	}
}
