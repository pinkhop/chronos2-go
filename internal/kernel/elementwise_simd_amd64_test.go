//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// simdReLUImplementations returns every amd64 SIMD implementation of ReLU
// the running CPU supports.
func simdReLUImplementations() []reluImplementation {
	if !archsimd.X86.AVX2() {
		return nil
	}
	return []reluImplementation{{name: "avx2", apply: avx2ReLU}}
}

// simdAddBiasImplementations returns every amd64 SIMD implementation of
// AddBias the running CPU supports.
func simdAddBiasImplementations() []addBiasImplementation {
	if !archsimd.X86.AVX2() {
		return nil
	}
	return []addBiasImplementation{{name: "avx2", apply: avx2AddBias}}
}

// simdAddBiasReLUImplementations returns every amd64 SIMD implementation of
// AddBiasReLU the running CPU supports.
func simdAddBiasReLUImplementations() []addBiasImplementation {
	if !archsimd.X86.AVX2() {
		return nil
	}
	return []addBiasImplementation{{name: "avx2", apply: avx2AddBiasReLU}}
}
