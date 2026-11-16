//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// simdSoftmaxRowsImplementations returns the NEON softmaxRows, which every
// arm64 CPU supports.
func simdSoftmaxRowsImplementations() []softmaxRowsImplementation {
	return []softmaxRowsImplementation{{name: "neon", normalize: softmaxRows}}
}

// vectorExpImplementations returns the NEON exponential, which every arm64
// CPU supports.
func vectorExpImplementations() []vectorExpImplementation {
	return []vectorExpImplementation{{name: "neon", exponentiate: neonExpAll}}
}

// neonExpAll evaluates neonExp on inputs four at a time, zero-padding the
// final chunk, and returns one result per input.
func neonExpAll(inputs []float32) []float32 {
	results := make([]float32, len(inputs))
	for i := 0; i < len(inputs); i += neonLanes {
		var chunk, result [neonLanes]float32
		copy(chunk[:], inputs[i:])
		neonExp(archsimd.LoadFloat32x4Array(&chunk)).StoreArray(&result)
		copy(results[i:], result[:])
	}
	return results
}
