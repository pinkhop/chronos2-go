//go:build goexperiment.simd

package kernel

// matMulTransposed is MatMulTransposed after its length checks. It runs the
// widest implementation the CPU supports, chosen once per process, so every
// call rounds identically.
func matMulTransposed(out, a, b []float32, m, k, n int) {
	switch detectedVectorLevel {
	case vectorLevelAVX512:
		avx512MatMulTransposed(out, a, b, m, k, n)
	case vectorLevelAVX2:
		avx2MatMulTransposed(out, a, b, m, k, n)
	case vectorLevelUnspecified, vectorLevelGeneric:
		matMulTransposedGeneric(out, a, b, m, k, n)
	}
}
