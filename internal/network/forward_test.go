package network

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

// nan32 returns a float32 NaN.
func nan32() float32 {
	return float32(math.NaN())
}

// TestForward_HiddenRequested_ReturnsFinalHiddenStates verifies that Forward returns the final hidden states when they are requested.
func TestForward_HiddenRequested_ReturnsFinalHiddenStates(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	fixture := syntheticFixture(t, "groups_gaps_covariates")

	// When
	output, err := network.Forward(t.Context(), batchFromFixture(fixture), ForwardOptions{Parallelism: 2, Hidden: true})
	// Then
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := fixture.Stages["hidden"]
	if output.Tokens != want.Shape[1] || output.Width != want.Shape[2] {
		t.Errorf("Tokens, Width = %d, %d; want %d, %d", output.Tokens, output.Width, want.Shape[1], want.Shape[2])
	}
	if len(output.Hidden) != len(want.Values) {
		t.Errorf("len(Hidden) = %d, want %d", len(output.Hidden), len(want.Values))
	}
}

// TestForward_HiddenNotRequested_LeavesHiddenNil verifies that Forward leaves Hidden nil when hidden states are not requested.
func TestForward_HiddenNotRequested_LeavesHiddenNil(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)

	// When
	output, err := network.Forward(t.Context(), singleRowBatch(20), ForwardOptions{Parallelism: 1})
	// Then
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Hidden != nil || output.Tokens != 0 || output.Width != 0 {
		t.Errorf("Hidden, Tokens, Width = %v, %d, %d; want nil, 0, 0", output.Hidden, output.Tokens, output.Width)
	}
}

// TestForward_ValidBatch_ReportsShapeLocationsAndScales verifies that Forward reports the output shape and each row's location and scale.
func TestForward_ValidBatch_ReportsShapeLocationsAndScales(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	fixture := syntheticFixture(t, "groups_gaps_covariates")

	// When
	output, err := network.Forward(t.Context(), batchFromFixture(fixture), ForwardOptions{Parallelism: 1})
	// Then
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Rows != 4 || output.Quantiles != 13 || output.Steps != 48 {
		t.Errorf("Rows, Quantiles, Steps = %d, %d, %d; want 4, 13, 48", output.Rows, output.Quantiles, output.Steps)
	}
	if len(output.Locations) != 4 || len(output.Scales) != 4 {
		t.Errorf("Locations, Scales = %v, %v; want 4 values each", output.Locations, output.Scales)
	}
}

// TestForward_DifferentParallelism_GivesIdenticalValues verifies that Forward gives identical values at every parallelism.
func TestForward_DifferentParallelism_GivesIdenticalValues(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	batch := batchFromFixture(syntheticFixture(t, "long"))
	serial, err := network.Forward(t.Context(), batch, ForwardOptions{Parallelism: 1})
	if err != nil {
		t.Fatalf("serial Forward: %v", err)
	}

	// When
	parallel, err := network.Forward(t.Context(), batch, ForwardOptions{Parallelism: 5})
	// Then
	if err != nil {
		t.Fatalf("parallel Forward: %v", err)
	}
	if !slices.Equal(serial.Values, parallel.Values) {
		t.Errorf("values differ between Parallelism 1 and 5")
	}
}

// TestForward_CanceledContext_ReturnsContextError verifies that Forward, given a canceled context, returns the context's error.
func TestForward_CanceledContext_ReturnsContextError(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	// When
	_, err := network.Forward(ctx, singleRowBatch(20), ForwardOptions{Parallelism: 1})

	// Then
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// TestForward_InvalidBatch_ReturnsErrInvalidBatch verifies that Forward returns ErrInvalidBatch for malformed batches.
func TestForward_InvalidBatch_ReturnsErrInvalidBatch(t *testing.T) {
	t.Parallel()
	cases := map[string]func(*Batch){
		"zero rows":              func(b *Batch) { b.Rows = 0 },
		"short context":          func(b *Batch) { b.Context = b.Context[:5] },
		"context too long":       func(b *Batch) { b.ContextLength = 9000; b.Context = make([]float32, 9000) },
		"missing group":          func(b *Batch) { b.GroupIDs = nil },
		"zero output patches":    func(b *Batch) { b.OutputPatches = 0; b.FutureCovariates = nil },
		"too many patches":       func(b *Batch) { b.OutputPatches = 65; b.FutureCovariates = make([]float32, 65*16) },
		"short future":           func(b *Batch) { b.FutureCovariates = b.FutureCovariates[:3] },
		"zero context length":    func(b *Batch) { b.ContextLength = 0; b.Context = nil },
		"rows disagree with ids": func(b *Batch) { b.GroupIDs = []int{0, 1} },
	}
	for name, corrupt := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Given
			network := syntheticNetwork(t)
			batch := singleRowBatch(20)
			corrupt(&batch)

			// When
			_, err := network.Forward(t.Context(), batch, ForwardOptions{Parallelism: 1})

			// Then
			if !errors.Is(err, ErrInvalidBatch) {
				t.Errorf("err = %v, want ErrInvalidBatch", err)
			}
		})
	}
}

// TestForward_TinyBlocks_MatchSingleBlockBitwise verifies that Forward with tiny work blocks matches a single block bit for bit.
func TestForward_TinyBlocks_MatchSingleBlockBitwise(t *testing.T) {
	t.Parallel()

	// Given
	network := syntheticNetwork(t)
	batch := batchFromFixture(syntheticFixture(t, "groups_gaps_covariates"))
	options := ForwardOptions{Parallelism: 3, Hidden: true}
	whole, err := network.Forward(t.Context(), batch, options)
	if err != nil {
		t.Fatalf("single-block Forward: %v", err)
	}

	// When
	blocked, err := network.forward(t.Context(), batch, options, nil, passTuning{blockTokens: 3})
	// Then
	if err != nil {
		t.Fatalf("blocked forward: %v", err)
	}
	if !slices.Equal(whole.Values, blocked.Values) || !slices.Equal(whole.Hidden, blocked.Hidden) {
		t.Errorf("tiny blocks changed the output")
	}
}

// TestForward_SingletonGroups_MatchGeneralGroupAttentionBitwise verifies that the singleton-group fast path matches general group attention bit for bit.
func TestForward_SingletonGroups_MatchGeneralGroupAttentionBitwise(t *testing.T) {
	t.Parallel()

	// Given: three rows in distinct groups, one of them with an unobserved
	// first patch so group attention hits both the self-only and the
	// all-masked case.
	network := syntheticNetwork(t)
	batch := singleRowBatch(40)
	batch.Rows, batch.GroupIDs = 3, []int{4, 9, 2}
	batch.Context = append(append(slices.Clone(batch.Context), batch.Context...), batch.Context...)
	for i := range 16 {
		batch.Context[40+i] = nan32()
	}
	batch.FutureCovariates = slices.Repeat(batch.FutureCovariates, 3)
	options := ForwardOptions{Parallelism: 2, Hidden: true}
	general, err := network.forward(t.Context(), batch, options, nil, passTuning{blockTokens: defaultBlockTokens, generalGroups: true})
	if err != nil {
		t.Fatalf("general forward: %v", err)
	}

	// When
	shortcut, err := network.Forward(t.Context(), batch, options)
	// Then
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !slices.Equal(general.Hidden, shortcut.Hidden) || !slices.Equal(general.Values, shortcut.Values) {
		t.Errorf("singleton-group shortcut changed the output")
	}
}
