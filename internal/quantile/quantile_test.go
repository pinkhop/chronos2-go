package quantile_test

import (
	"math"
	"testing"
)

// tolerance is the absolute difference allowed between a float32 computation
// and its hand-computed value; inputs such as 0.1 are not exact in binary.
const tolerance = 1e-6

// assertClose fails the test for each element of got farther than tolerance
// from want, or when the lengths differ. NaN matches only NaN.
func assertClose[T float32 | float64](t *testing.T, label string, got, want []T) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		gotValue, wantValue := float64(got[i]), float64(want[i])
		if math.IsNaN(gotValue) && math.IsNaN(wantValue) {
			continue
		}
		if math.Abs(gotValue-wantValue) > tolerance {
			t.Errorf("%s[%d] = %v, want %v", label, i, got[i], want[i])
		}
	}
}

// panicValue calls f and returns the value it panicked with, or nil.
func panicValue(f func()) (value any) {
	defer func() { value = recover() }()
	f()
	return nil
}
