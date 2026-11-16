package kernel

// Dot returns the inner product of a and b, accumulated in float32 as
// PyTorch does on the CPU.
//
// It panics when a and b differ in length. Both may be empty, giving 0.
func Dot(a, b []float32) float32 {
	clearUpperBits()
	requireLength("b", len(b), len(a))
	b = b[:len(a)]

	// Four independent accumulators shorten the floating-point dependency
	// chain; the summation order differs from a strict left-to-right sum only
	// within float32 rounding.
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		sum0 += a[i] * b[i]
	}
	return (sum0 + sum1) + (sum2 + sum3)
}
