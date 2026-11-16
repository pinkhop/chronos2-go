package kernel

// MatMulTransposed computes out = a · bᵀ, where a is a row-major [m × k]
// matrix, b is a row-major [n × k] matrix, and out is a row-major [m × n]
// matrix. It matches PyTorch's [out, in] linear-weight layout, so b is a
// weight read without transposing, and it also computes attention scores
// Q · Kᵀ.
//
// Each output is accumulated in float32 by one fixed procedure that depends
// only on k, so a row's result does not depend on m or on which rows share a
// call; callers may split m across goroutines without changing any value.
// The generic and SIMD builds sum in different orders and agree within
// float32 rounding, not bitwise. out is fully overwritten and must not
// overlap a or b. It panics when a length disagrees with m, k, and n.
func MatMulTransposed(out, a, b []float32, m, k, n int) {
	requireLength("a", len(a), m*k)
	requireLength("b", len(b), n*k)
	requireLength("out", len(out), m*n)
	matMulTransposed(out, a, b, m, k, n)
}
