//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx2Lanes is the number of float32 lanes in an AVX2 vector register.
const avx2Lanes = 8

// avx2SumLanes is the number of sums avx2ReduceFour packs into one 128-bit
// result.
const avx2SumLanes = 4

// avx2ReduceFour returns the lane sums of c0, c1, c2, and c3 in lanes 0
// through 3, each reduced as ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)). It is the
// only lane reduction the AVX2 kernels use, which keeps every output's
// rounding identical across paths. The two horizontal adds work within each
// 128-bit half, leaving the low-half sums of c0 through c3 in the low half and
// the high-half sums in the high half.
func avx2ReduceFour(c0, c1, c2, c3 archsimd.Float32x8) archsimd.Float32x4 {
	halves := c0.ConcatAddPairsGrouped(c1).ConcatAddPairsGrouped(c2.ConcatAddPairsGrouped(c3))
	return halves.GetLo().Add(halves.GetHi())
}

// avx2LoadFull returns s[p:p+8] as a vector; p+8 must not exceed len(s).
func avx2LoadFull(s []float32, p int) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[avx2Lanes]float32)(s[p : p+avx2Lanes]))
}

// avx2TailMask holds eight zeros then eight all-ones lanes. The eight
// lanes starting at index r keep the last r lanes of a vector and zero the
// rest; it is never modified.
var avx2TailMask = [2 * avx2Lanes]int32{0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1}

// avx2LoadPart returns the final partial chunk s[p:], which holds fewer than
// eight values, as a vector whose other lanes are exact zeros, so they add
// nothing to a fused multiply-add. When s holds at least eight values, it
// loads the last eight and zeroes the lanes before p, which the full chunks
// already counted, so the chunk's values occupy the high lanes; a shorter s
// is copied into the low lanes of a zeroed array. Either way the lane
// placement depends only on len(s) and p, so every output computed from
// slices of one length uses one procedure. p must be less than len(s).
//
// It never calls archsimd.LoadFloat32x8Part, which in Go 1.27 loads a full
// 32 bytes and masks them, reading past the end of s; past the last
// embedded weight that read could cross into an unmapped page.
func avx2LoadPart(s []float32, p int) archsimd.Float32x8 {
	if len(s) < avx2Lanes {
		return avx2LoadPadded(s, p, 0)
	}
	remaining := len(s) - p
	keep := archsimd.LoadInt32x8Array((*[avx2Lanes]int32)(avx2TailMask[remaining : remaining+avx2Lanes]))
	return avx2LoadFull(s, len(s)-avx2Lanes).AsInt32x8().And(keep).AsFloat32x8()
}

// avx2LoadPadded returns the final partial chunk s[p:] as a vector with its
// missing lanes set to fill. p must be less than len(s).
func avx2LoadPadded(s []float32, p int, fill float32) archsimd.Float32x8 {
	chunk := [avx2Lanes]float32{fill, fill, fill, fill, fill, fill, fill, fill}
	// An explicit loop, unlike copy, compiles without a call to memmove, so
	// the caller's accumulators need not be spilled around it.
	for i, value := range s[p:min(len(s), p+avx2Lanes)] {
		chunk[i] = value
	}
	return archsimd.LoadFloat32x8Array(&chunk)
}
