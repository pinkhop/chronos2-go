package network

import (
	"math"
	"slices"
	"testing"
)

// edgeBatch builds a batch of rows rows, each of length steps (a repeating
// ramp), in groups, predicting patches output patches with unknown futures.
func edgeBatch(rows, length, patches int, groups []int) Batch {
	context := make([]float32, rows*length)
	for i := range context {
		context[i] = float32(i%11) - 3
	}
	return Batch{
		Rows: rows, ContextLength: length, Context: context, GroupIDs: groups,
		OutputPatches: patches, FutureCovariates: slices.Repeat([]float32{nan32()}, rows*patches*16),
	}
}

// edgeBatches returns batches at the boundaries the pass's patching,
// masking, and blocking must handle, keyed by description.
func edgeBatches(config Config) map[string]Batch {
	allMissing := edgeBatch(3, 33, 1, []int{0, 0, 1})
	for i := range 33 {
		allMissing.Context[33+i] = nan32()
	}
	return map[string]Batch{
		"one step":                 edgeBatch(1, 1, 1, []int{0}),
		"one step short of patch":  edgeBatch(1, 15, 1, []int{0}),
		"exactly one patch":        edgeBatch(1, 16, 1, []int{0}),
		"one step past one patch":  edgeBatch(1, 17, 1, []int{0}),
		"exactly two patches":      edgeBatch(2, 32, 2, []int{0, 1}),
		"one step past two":        edgeBatch(2, 33, 2, []int{0, 0}),
		"row with no observations": allMissing,
		"max output patches":       edgeBatch(1, 20, config.MaxOutputPatches, []int{0}),
		"more rows than a block":   edgeBatch(7, 24, 1, []int{0, 0, 1, 2, 2, 2, 3}),
	}
}

// allFinite reports whether values holds no NaN or infinity.
func allFinite(values []float32) bool {
	return !slices.ContainsFunc(values, func(value float32) bool {
		return math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
	})
}

// TestForward_EdgeBatches_TinyBlocksMatchDefaultAndStayFinite verifies that,
// for batches at patch, mask, and block boundaries, three-token blocks
// reproduce the default blocking bit for bit, hidden states included, and
// every forecast is finite.
func TestForward_EdgeBatches_TinyBlocksMatchDefaultAndStayFinite(t *testing.T) {
	t.Parallel()

	network := syntheticNetwork(t)
	for name, batch := range edgeBatches(network.config) {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Given the default blocking's output
			options := ForwardOptions{Parallelism: 3, Hidden: true}
			whole, err := network.Forward(t.Context(), batch, options)
			if err != nil {
				t.Fatalf("default Forward: %v", err)
			}

			// When the same batch runs in three-token blocks
			blocked, err := network.forward(t.Context(), batch, options, nil, passTuning{blockTokens: 3})
			// Then nothing changes and every forecast is finite
			if err != nil {
				t.Fatalf("blocked forward: %v", err)
			}
			if !slices.Equal(whole.Values, blocked.Values) || !slices.Equal(whole.Hidden, blocked.Hidden) {
				t.Errorf("tiny blocks changed the output")
			}
			if !allFinite(blocked.Values) {
				t.Errorf("forecast holds NaN or infinity")
			}
		})
	}
}
