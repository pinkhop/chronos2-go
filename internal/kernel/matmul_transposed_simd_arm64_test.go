//go:build goexperiment.simd

package kernel

// simdMatMulTransposedImplementations returns the NEON matMulTransposed,
// which every arm64 CPU supports.
func simdMatMulTransposedImplementations() []matMulTransposedImplementation {
	return []matMulTransposedImplementation{{name: "neon", multiply: matMulTransposed}}
}
