//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// addBias is AddBias after its length checks. It uses AVX2 on CPUs with AVX2
// and FMA, including those with AVX-512, since the pass is bound by memory
// rather than arithmetic.
func addBias(x, bias []float32, rows, cols int) {
	switch detectedVectorLevel {
	case vectorLevelAVX2, vectorLevelAVX512:
		avx2AddBias(x, bias, rows, cols)
	case vectorLevelUnspecified, vectorLevelGeneric:
		addBiasGeneric(x, bias, rows, cols)
	}
}

// avx2AddBias is AddBias using AVX2 instructions, eight elements at a time,
// with the elements after each row's last full vector handled by
// addBiasRowGeneric. Each element gets the same single float32 addition as
// in the generic code. The CPU must support AVX2. Like every amd64 vector
// kernel, it clears the upper vector-register bits before returning (see
// clearUpperBits).
func avx2AddBias(x, bias []float32, rows, cols int) {
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		c := 0
		for ; c+avx2Lanes <= cols; c += avx2Lanes {
			sum := avx2LoadFull(row, c).Add(avx2LoadFull(bias, c))
			sum.StoreArray((*[avx2Lanes]float32)(row[c : c+avx2Lanes]))
		}
		addBiasRowGeneric(row[c:], bias[c:])
	}
	archsimd.ClearAVXUpperBits()
}
