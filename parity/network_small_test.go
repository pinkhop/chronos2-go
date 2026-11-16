package parity

import "testing"

// TestNetwork_SmallModelStages_MatchReference verifies that the small model's forward stages match the reference's intermediate values.
func TestNetwork_SmallModelStages_MatchReference(t *testing.T) {
	t.Parallel()
	net := realNetwork(t, "small")
	for _, scenario := range forwardScenarios {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			checkForwardStages(t, net, "small", scenario)
		})
	}
}
