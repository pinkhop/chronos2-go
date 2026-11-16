package quantile

import "fmt"

// LevelMass returns the probability mass each quantile level represents,
// normalized to sum to 1. The long-horizon unroll weights its sample paths
// by the outer product of two such mass vectors.
//
// It reproduces the reference's trapezoidal rule: level i receives
// (levels[i+1] − levels[i−1]) / 2, taking levels[−1] as 0 and levels[n] as 1,
// computed in float32 and then normalized. levels should ascend. It panics
// when levels is empty or any level lies outside the open interval (0, 1).
// LevelMass does not modify levels and is safe for concurrent use.
func LevelMass(levels []float64) []float64 {
	if len(levels) == 0 {
		panic("quantile: LevelMass: no levels")
	}

	// Surround the levels with the boundaries 0 and 1.
	boundaries := make([]float32, 0, len(levels)+2)
	boundaries = append(boundaries, 0)
	for _, level := range levels {
		if !(level > 0 && level < 1) {
			panic(fmt.Sprintf("quantile: LevelMass: level %v outside (0, 1)", level))
		}
		boundaries = append(boundaries, float32(level))
	}
	boundaries = append(boundaries, 1)

	// Take each level's half-width between its neighbors, then normalize.
	masses := make([]float32, len(levels))
	var total float64
	for i := range masses {
		masses[i] = (boundaries[i+2] - boundaries[i]) / 2
		total += float64(masses[i])
	}
	total32 := float32(total)
	normalized := make([]float64, len(masses))
	for i, mass := range masses {
		normalized[i] = float64(mass / total32)
	}
	return normalized
}
