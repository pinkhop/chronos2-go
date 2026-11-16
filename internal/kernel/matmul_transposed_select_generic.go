//go:build !goexperiment.simd || !(arm64 || amd64)

package kernel

// matMulTransposed is MatMulTransposed after its length checks. Builds without
// a SIMD implementation for the target architecture use the generic code.
func matMulTransposed(out, a, b []float32, m, k, n int) {
	matMulTransposedGeneric(out, a, b, m, k, n)
}
