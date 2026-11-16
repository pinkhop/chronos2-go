//go:build slow

package parity

import (
	"testing"

	base "github.com/pinkhop/chronos2-go-base"
	small "github.com/pinkhop/chronos2-go-small"
)

// TestSmallModel_SlowTierScenarios_MatchReference verifies that the small model matches the reference on the slow-tier scenarios.
func TestSmallModel_SlowTierScenarios_MatchReference(t *testing.T) {
	t.Parallel()
	checkScenarios(t, "small", small.New, true)
}

// TestBaseModel_DefaultTierScenarios_MatchReference verifies that the base model matches the reference on the default-tier scenarios.
func TestBaseModel_DefaultTierScenarios_MatchReference(t *testing.T) {
	t.Parallel()
	checkScenarios(t, "base", base.New, false)
}

// TestBaseModel_SlowTierScenarios_MatchReference verifies that the base model matches the reference on the slow-tier scenarios.
func TestBaseModel_SlowTierScenarios_MatchReference(t *testing.T) {
	t.Parallel()
	checkScenarios(t, "base", base.New, true)
}
