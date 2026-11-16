package kernel

// ReLU replaces every negative element of x with zero, in place. NaN
// elements stay NaN, as in PyTorch, and negative zero becomes positive zero.
// The generic and SIMD builds give bitwise-identical results.
func ReLU(x []float32) {
	relu(x)
}
