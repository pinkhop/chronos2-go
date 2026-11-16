package testmodel

import (
	"math"
	"path/filepath"
	"testing"
)

// TestReadScenario_UnivariateShort_DecodesInputsAndOutputs verifies that ReadScenario decodes a short univariate scenario's inputs and outputs.
func TestReadScenario_UnivariateShort_DecodesInputsAndOutputs(t *testing.T) {
	t.Parallel()

	// Given: the synthetic univariate scenario.
	path := filepath.Join(SyntheticDir(), "scenarios", "univariate_short.json")

	// When: it is read.
	scenario, outputs := ReadScenario(t, path)

	// Then: inputs and expected outputs have the recorded shapes.
	if scenario.Method != "Forecast" || scenario.Horizon != 24 || len(scenario.Tasks) != 1 {
		t.Fatalf("scenario = %+v, want one Forecast task with horizon 24", scenario)
	}
	if got := len(scenario.Tasks[0].Targets[0]); got != 100 {
		t.Errorf("history length = %d, want 100", got)
	}
	if got := outputs["task_000"].Shape; len(got) != 3 || got[0] != 1 || got[1] != 13 || got[2] != 24 {
		t.Errorf("task_000 shape = %v, want [1 13 24]", got)
	}
}

// TestReadScenario_Gaps_MapsNullToNaN verifies that ReadScenario, given gaps, maps null to NaN.
func TestReadScenario_Gaps_MapsNullToNaN(t *testing.T) {
	t.Parallel()

	// Given: the synthetic scenario whose history has gaps.
	path := filepath.Join(SyntheticDir(), "scenarios", "gaps.json")

	// When: it is read.
	scenario, _ := ReadScenario(t, path)

	// Then: the recorded gaps decode as NaN.
	if history := scenario.Tasks[0].Targets[0]; !math.IsNaN(history[5]) || math.IsNaN(history[0]) {
		t.Errorf("history[0]=%v history[5]=%v, want a value then NaN", history[0], history[5])
	}
}

// TestSyntheticCheckpoint_Files_AreReadable verifies that the synthetic checkpoint's files are readable.
func TestSyntheticCheckpoint_Files_AreReadable(t *testing.T) {
	t.Parallel()

	// Given/When: the synthetic checkpoint is read.
	config, shards := SyntheticCheckpoint(t)

	// Then: one shard and a non-empty config are returned.
	if len(config) == 0 || len(shards) != 1 || len(shards[0]) == 0 {
		t.Errorf("config %d bytes, %d shards; want a config and one shard", len(config), len(shards))
	}
}

// TestWithin_CombinedTolerance_AcceptsAndRejects verifies that Within accepts and rejects values against a combined tolerance.
func TestWithin_CombinedTolerance_AcceptsAndRejects(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		got, want float64
		ok        bool
	}{
		{name: "inside absolute", got: 1.00005, want: 1, ok: true},
		{name: "inside relative", got: 1001, want: 1000, ok: true},
		{name: "outside", got: 1.01, want: 1, ok: false},
		{name: "both NaN", got: math.NaN(), want: math.NaN(), ok: true},
		{name: "one NaN", got: math.NaN(), want: 1, ok: false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()

			// When: the values are compared with abs 1e-4 and rel 1e-3.
			ok := Within(c.got, c.want, 1e-4, 1e-3)

			// Then: the verdict matches.
			if ok != c.ok {
				t.Errorf("Within(%v, %v) = %v, want %v", c.got, c.want, ok, c.ok)
			}
		})
	}
}
