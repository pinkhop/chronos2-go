//go:build slow

package parity

import "testing"

// TestNetwork_BaseModelStages_MatchReference verifies that the base model's forward stages match the reference's intermediate values.
func TestNetwork_BaseModelStages_MatchReference(t *testing.T) {
	t.Parallel()
	net := realNetwork(t, "base")
	for _, scenario := range forwardScenarios {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			checkForwardStages(t, net, "base", scenario)
		})
	}
}
