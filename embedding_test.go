package chronos2

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// TestModel_Embed_HistoryOf20Steps_ReturnsPatchRegAndFutureStates verifies that Model.Embed, given a 20-step history, returns patch, [REG], and future states.
func TestModel_Embed_HistoryOf20Steps_ReturnsPatchRegAndFutureStates(t *testing.T) {
	t.Parallel()

	// Given 20 steps: two patches, then [REG], then one output patch.
	model, forward, _ := newFakeModel()

	// When
	embedding, err := model.Embed(t.Context(), ramp(1, 20))
	// Then
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if batch, options := forward.batches[0], forward.options[0]; batch.OutputPatches != 1 || !options.hidden || len(batch.FutureCovariates) != 16 {
		t.Errorf("pass = %d patches, hidden %v, %d future values; want 1, true, 16", batch.OutputPatches, options.hidden, len(batch.FutureCovariates))
	}
	if len(embedding.states) != 4 || embedding.patches != 2 || embedding.summary != 2 {
		t.Errorf("embedding has %d states, %d patches, summary %d; want 4, 2, 2", len(embedding.states), embedding.patches, embedding.summary)
	}
	if embedding.location != 20 || embedding.scale != 2 {
		t.Errorf("location, scale = %v, %v; want 20, 2", embedding.location, embedding.scale)
	}
}

// TestEmbedding_Summary_RegState_ReturnsRegVector verifies that Embedding.Summary returns the [REG] state's vector.
func TestEmbedding_Summary_RegState_ReturnsRegVector(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()
	embedding, err := model.Embed(t.Context(), ramp(1, 20))
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	// When
	summary := embedding.Summary()

	// Then
	if !slices.Equal(summary, []float32{2000, 2001, 2002, 2003}) {
		t.Errorf("Summary() = %v, want token 2's vector", summary)
	}
}

// TestEmbedding_Patches_TwoPatches_ReturnsOldestFirst verifies that Embedding.Patches, given two patches, returns oldest first.
func TestEmbedding_Patches_TwoPatches_ReturnsOldestFirst(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()
	embedding, err := model.Embed(t.Context(), ramp(1, 20))
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	// When
	patches := embedding.Patches()

	// Then
	if len(patches) != 2 || patches[0][0] != 0 || patches[1][0] != 1000 {
		t.Errorf("Patches() = %v, want tokens 0 and 1", patches)
	}
}

// TestEmbedding_States_CallerModifiesResult_EmbeddingUnchanged verifies that Embedding.States, given caller modifies result, leaves embedding unchanged.
func TestEmbedding_States_CallerModifiesResult_EmbeddingUnchanged(t *testing.T) {
	t.Parallel()

	// Given
	model, _, _ := newFakeModel()
	embedding, err := model.Embed(t.Context(), ramp(1, 20))
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	embedding.States()[0][0] = 999

	// When
	states := embedding.States()

	// Then
	if len(states) != 4 || states[0][0] != 0 {
		t.Errorf("States() = %v after modifying an earlier result, want token 0 to start with 0", states)
	}
}

// TestModel_Embed_ContextLengthOption_TrimsHistory verifies that Model.Embed, given context length option, trims history.
func TestModel_Embed_ContextLengthOption_TrimsHistory(t *testing.T) {
	t.Parallel()

	// Given
	model, forward, _ := newFakeModel()

	// When
	embedding, err := model.Embed(t.Context(), ramp(1, 50), WithContextLength(17))
	// Then
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if forward.batches[0].ContextLength != 17 || embedding.patches != 2 {
		t.Errorf("context = %d steps, %d patches; want 17 and 2", forward.batches[0].ContextLength, embedding.patches)
	}
}

// TestModel_Embed_InvalidInput_ReturnsSentinel verifies that Model.Embed, given invalid input, returns sentinel.
func TestModel_Embed_InvalidInput_ReturnsSentinel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		history []float64
		options []EmbedOption
		want    error
	}{
		{"empty", nil, nil, ErrEmptyHistory},
		{"all NaN", []float64{math.NaN()}, nil, ErrNoObservations},
		{"infinite", []float64{1, math.Inf(-1)}, nil, ErrInvalidValue},
		{"beyond float32", []float64{1, math.MaxFloat64}, nil, ErrInvalidValue},
		{"bad context length", ramp(1, 4), []EmbedOption{WithContextLength(100)}, ErrInvalidOption},
		{"bad parallelism", ramp(1, 4), []EmbedOption{WithMaxParallelism(-1)}, ErrInvalidOption},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given
			model, _, _ := newFakeModel()

			// When
			_, err := model.Embed(t.Context(), tt.history, tt.options...)

			// Then
			if !errors.Is(err, tt.want) {
				t.Errorf("Embed() error = %v, want %v", err, tt.want)
			}
		})
	}
}
