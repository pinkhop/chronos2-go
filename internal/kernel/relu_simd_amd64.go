//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// relu is ReLU. It uses AVX2 on CPUs with AVX2 and FMA, including those with
// AVX-512, since the pass is bound by memory rather than arithmetic.
func relu(x []float32) {
	switch detectedVectorLevel {
	case vectorLevelAVX2, vectorLevelAVX512:
		avx2ReLU(x)
	case vectorLevelUnspecified, vectorLevelGeneric:
		reluGeneric(x)
	}
}

// avx2ReLU is ReLU using AVX2 instructions, eight elements at a time, with
// the elements after the last full vector handled by reluGeneric. It keeps a
// lane where the value is positive or NaN and writes +0 elsewhere, which is
// exactly the built-in max(value, 0) the generic code uses; x86's maximum
// instruction alone would differ for NaN and negative zero. The CPU must
// support AVX2. It clears the upper vector-register bits before the scalar
// tail (see clearUpperBits).
func avx2ReLU(x []float32) {
	var zero archsimd.Float32x8
	p := 0
	for ; p+avx2Lanes <= len(x); p += avx2Lanes {
		value := avx2LoadFull(x, p)
		keep := value.Greater(zero).Or(value.IsNaN())
		value.Masked(keep).StoreArray((*[avx2Lanes]float32)(x[p : p+avx2Lanes]))
	}
	archsimd.ClearAVXUpperBits()
	reluGeneric(x[p:])
}
