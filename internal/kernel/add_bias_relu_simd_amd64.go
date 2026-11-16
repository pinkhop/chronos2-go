//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// addBiasReLU is AddBiasReLU after its length checks. It uses AVX2 on CPUs
// with AVX2 and FMA, including those with AVX-512, since the pass is bound by
// memory rather than arithmetic.
func addBiasReLU(x, bias []float32, rows, cols int) {
	switch detectedVectorLevel {
	case vectorLevelAVX2, vectorLevelAVX512:
		avx2AddBiasReLU(x, bias, rows, cols)
	case vectorLevelUnspecified, vectorLevelGeneric:
		addBiasReLUGeneric(x, bias, rows, cols)
	}
}

// avx2AddBiasReLU is AddBiasReLU using AVX2 instructions, eight elements at
// a time, with the elements after each row's last full vector handled by
// addBiasReLURowGeneric. Each sum is the same single float32 addition as in
// the generic code, and the clamp keeps exactly the lanes avx2ReLU keeps. The
// CPU must support AVX2. It clears the upper vector-register bits before each
// scalar row tail (see clearUpperBits).
func avx2AddBiasReLU(x, bias []float32, rows, cols int) {
	var zero archsimd.Float32x8
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		c := 0
		for ; c+avx2Lanes <= cols; c += avx2Lanes {
			sum := avx2LoadFull(row, c).Add(avx2LoadFull(bias, c))
			keep := sum.Greater(zero).Or(sum.IsNaN())
			sum.Masked(keep).StoreArray((*[avx2Lanes]float32)(row[c : c+avx2Lanes]))
		}
		archsimd.ClearAVXUpperBits()
		addBiasReLURowGeneric(row[c:], bias[c:])
	}
}
