package kernel

// reluGeneric is the portable implementation of ReLU, compiled into every
// build; the SIMD builds use it for the elements after their last full
// vector, and their tests compare against it.
func reluGeneric(x []float32) {
	// The built-in max is branch-free and returns NaN when either operand is
	// NaN, which keeps random signs from stalling on mispredicted branches.
	for i, value := range x {
		x[i] = max(value, 0)
	}
}
