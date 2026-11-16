//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// simdSoftmaxRowsImplementations returns every amd64 SIMD implementation of
// softmaxRows the running CPU supports.
func simdSoftmaxRowsImplementations() []softmaxRowsImplementation {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() {
		return nil
	}
	return []softmaxRowsImplementation{{name: "avx2", normalize: avx2SoftmaxRows}}
}

// vectorExpImplementations returns every amd64 vector exponential the
// running CPU supports.
func vectorExpImplementations() []vectorExpImplementation {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() {
		return nil
	}
	return []vectorExpImplementation{{name: "avx2", exponentiate: avx2ExpAll}}
}

// avx2ExpAll evaluates avx2Exp on inputs eight at a time, zero-padding the
// final chunk, and returns one result per input.
func avx2ExpAll(inputs []float32) []float32 {
	results := make([]float32, len(inputs))
	for i := 0; i < len(inputs); i += avx2Lanes {
		var chunk, result [avx2Lanes]float32
		copy(chunk[:], inputs[i:])
		avx2Exp(archsimd.LoadFloat32x8Array(&chunk)).StoreArray(&result)
		copy(results[i:], result[:])
	}
	return results
}
