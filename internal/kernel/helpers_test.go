package kernel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// relativeTolerance is the largest acceptable relative error between a kernel
// and its float64 reference; it allows for float32 rounding and summation
// order while catching any indexing or formula error.
const relativeTolerance = 1e-5

// absoluteTolerance is the error floor applied to values near zero, where a
// relative bound is meaningless.
const absoluteTolerance = 1e-5

// randomValues returns count deterministic pseudo-random values in [-1, 1),
// seeded by seed, so tests and benchmarks are reproducible.
func randomValues(seed uint64, count int) []float32 {
	generator := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	values := make([]float32, count)
	for i := range values {
		values[i] = generator.Float32()*2 - 1
	}
	return values
}

// assertClose fails the test when any element of got differs from want by
// more than the combined absolute and relative tolerance; label names the
// quantity in failure messages.
func assertClose(t *testing.T, label string, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", label, len(got), len(want))
	}
	for i := range got {
		difference := math.Abs(float64(got[i]) - want[i])
		if difference > absoluteTolerance+relativeTolerance*math.Abs(want[i]) {
			t.Errorf("%s[%d] = %g, want %g", label, i, got[i], want[i])
			return
		}
	}
}

// assertPanics fails the test unless call panics; it lets length-mismatch
// tests keep a single When step.
func assertPanics(t *testing.T, call func()) {
	t.Helper()
	panicked := func() (recovered bool) {
		defer func() { recovered = recover() != nil }()
		call()
		return false
	}()
	if !panicked {
		t.Errorf("call did not panic")
	}
}
