//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx512Lanes is the number of float32 lanes in an AVX-512 vector register.
const avx512Lanes = 16

// avx512ReduceFour returns the lane sums of c0, c1, c2, and c3 in lanes 0
// through 3. Each accumulator first folds its high eight lanes onto its low
// eight (lane i plus lane i+8), and avx2ReduceFour then sums those eight. It
// is the only lane reduction the AVX-512 kernels use, which keeps every
// output's rounding identical across paths.
func avx512ReduceFour(c0, c1, c2, c3 archsimd.Float32x16) archsimd.Float32x4 {
	return avx2ReduceFour(
		c0.GetLo().Add(c0.GetHi()), c1.GetLo().Add(c1.GetHi()),
		c2.GetLo().Add(c2.GetHi()), c3.GetLo().Add(c3.GetHi()))
}

// avx512LoadFull returns s[p:p+16] as a vector; p+16 must not exceed len(s).
func avx512LoadFull(s []float32, p int) archsimd.Float32x16 {
	return archsimd.LoadFloat32x16Array((*[avx512Lanes]float32)(s[p : p+avx512Lanes]))
}

// avx512LoadPart returns the final partial chunk s[p:], which holds fewer
// than sixteen values, as a vector whose other lanes are exact zeros. Like
// avx2LoadPart, it loads the last sixteen values and zeroes the lanes before
// p when s holds at least sixteen, and otherwise copies s[p:] into the low
// lanes of a zeroed array; it never calls archsimd.LoadFloat32x16Part, which
// reads past the end of s. p must be less than len(s).
func avx512LoadPart(s []float32, p int) archsimd.Float32x16 {
	if len(s) >= avx512Lanes {
		remaining := len(s) - p
		keep := archsimd.Mask32x16FromBits(uint16(0xffff << (avx512Lanes - remaining)))
		return avx512LoadFull(s, len(s)-avx512Lanes).Masked(keep)
	}
	var chunk [avx512Lanes]float32
	// An explicit loop, unlike copy, compiles without a call to memmove, so
	// the caller's accumulators need not be spilled around it.
	//lint:ignore S1001 copy would call memmove and spill the accumulators.
	for i, value := range s[p:] {
		chunk[i] = value
	}
	return archsimd.LoadFloat32x16Array(&chunk)
}
