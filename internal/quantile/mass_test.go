package quantile_test

import (
	"testing"

	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// TestLevelMass_UnevenLevels_ReturnsNormalizedTrapezoidMass verifies that LevelMass, given uneven levels, returns normalized trapezoid mass.
func TestLevelMass_UnevenLevels_ReturnsNormalizedTrapezoidMass(t *testing.T) {
	t.Parallel()

	// Given levels whose raw masses are 0.25, 0.4, and 0.25, summing to 0.9.
	levels := []float64{0.1, 0.5, 0.9}

	// When
	got := quantile.LevelMass(levels)

	// Then
	assertClose(t, "mass", got, []float64{0.25 / 0.9, 0.4 / 0.9, 0.25 / 0.9})
}

// TestLevelMass_Deciles_ReturnsUniformMass verifies that LevelMass, given deciles, returns uniform mass.
func TestLevelMass_Deciles_ReturnsUniformMass(t *testing.T) {
	t.Parallel()

	// Given
	levels := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	// When
	got := quantile.LevelMass(levels)

	// Then
	want := make([]float64, len(levels))
	for i := range want {
		want[i] = 1.0 / 9
	}
	assertClose(t, "mass", got, want)
}

// TestLevelMass_LevelOutsideOpenInterval_Panics verifies that LevelMass, given level outside open interval, panics.
func TestLevelMass_LevelOutsideOpenInterval_Panics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		levels []float64
	}{
		{"empty", nil},
		{"zero", []float64{0, 0.5}},
		{"one", []float64{0.5, 1}},
		{"NaN", []float64{nan()}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt.levels

			// When
			recovered := panicValue(func() { quantile.LevelMass(tt.levels) })

			// Then
			if recovered == nil {
				t.Errorf("LevelMass() did not panic")
			}
		})
	}
}
