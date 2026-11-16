package parity

import (
	"testing"

	small "github.com/pinkhop/chronos2-go-small"
)

// TestSmallModel_DefaultTierScenarios_MatchReference verifies that the small model matches the reference on the default-tier scenarios.
func TestSmallModel_DefaultTierScenarios_MatchReference(t *testing.T) {
	t.Parallel()
	checkScenarios(t, "small", small.New, false)
}
