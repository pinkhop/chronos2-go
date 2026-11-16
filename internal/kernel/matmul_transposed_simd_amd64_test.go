//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// simdMatMulTransposedImplementations returns every amd64 SIMD
// implementation of matMulTransposed the running CPU supports, so a CPU with
// AVX-512 tests its AVX2 path too.
func simdMatMulTransposedImplementations() []matMulTransposedImplementation {
	var implementations []matMulTransposedImplementation
	if archsimd.X86.AVX2() && archsimd.X86.FMA() {
		implementations = append(implementations, matMulTransposedImplementation{name: "avx2", multiply: avx2MatMulTransposed})
	}
	if archsimd.X86.AVX512() && archsimd.X86.FMA() {
		implementations = append(implementations, matMulTransposedImplementation{name: "avx512", multiply: avx512MatMulTransposed})
	}
	return implementations
}
