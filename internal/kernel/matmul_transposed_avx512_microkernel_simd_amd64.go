//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx512ThreeByFour returns the twelve inner products of a0, a1, and a2
// with b0 through b3. Result R holds aR with b0 through b3 in lanes 0 through
// 3, ready to store as four adjacent outputs. Each chunk of b is loaded once
// and used at once by all three rows, the last of which takes it as the
// receiver; the last column makes each chunk of a the receiver at its last
// use instead (see the note on VFMADD213PS in the AVX2 micro-kernels). All
// slices share a length.
func avx512ThreeByFour(a0, a1, a2, b0, b1, b2, b3 []float32) (r0, r1, r2 archsimd.Float32x4) {
	// Reslicing to a common length lets the compiler drop bounds checks.
	a1, a2 = a1[:len(a0)], a2[:len(a0)]
	b0, b1, b2, b3 = b0[:len(a0)], b1[:len(a0)], b2[:len(a0)], b3[:len(a0)]

	var c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23 archsimd.Float32x16
	p := 0
	for ; p+avx512Lanes <= len(a0); p += avx512Lanes {
		x0, x1, x2 := avx512LoadFull(a0, p), avx512LoadFull(a1, p), avx512LoadFull(a2, p)
		y := avx512LoadFull(b0, p)
		c00, c10, c20 = x0.MulAdd(y, c00), x1.MulAdd(y, c10), y.MulAdd(x2, c20)
		y = avx512LoadFull(b1, p)
		c01, c11, c21 = x0.MulAdd(y, c01), x1.MulAdd(y, c11), y.MulAdd(x2, c21)
		y = avx512LoadFull(b2, p)
		c02, c12, c22 = x0.MulAdd(y, c02), x1.MulAdd(y, c12), y.MulAdd(x2, c22)
		y = avx512LoadFull(b3, p)
		c03, c13, c23 = y.MulAdd(x0, c03), y.MulAdd(x1, c13), y.MulAdd(x2, c23)
	}
	if p < len(a0) {
		x0, x1, x2 := avx512LoadPart(a0, p), avx512LoadPart(a1, p), avx512LoadPart(a2, p)
		y := avx512LoadPart(b0, p)
		c00, c10, c20 = x0.MulAdd(y, c00), x1.MulAdd(y, c10), y.MulAdd(x2, c20)
		y = avx512LoadPart(b1, p)
		c01, c11, c21 = x0.MulAdd(y, c01), x1.MulAdd(y, c11), y.MulAdd(x2, c21)
		y = avx512LoadPart(b2, p)
		c02, c12, c22 = x0.MulAdd(y, c02), x1.MulAdd(y, c12), y.MulAdd(x2, c22)
		y = avx512LoadPart(b3, p)
		c03, c13, c23 = y.MulAdd(x0, c03), y.MulAdd(x1, c13), y.MulAdd(x2, c23)
	}
	return avx512ReduceFour(c00, c01, c02, c03), avx512ReduceFour(c10, c11, c12, c13),
		avx512ReduceFour(c20, c21, c22, c23)
}

// avx512FourByOne returns the inner products of a0, a1, a2, and a3 with b in
// lanes 0 through 3. Because the inner product is symmetric, it also serves
// one row of a against four rows of b. All slices share a length.
func avx512FourByOne(a0, a1, a2, a3, b []float32) archsimd.Float32x4 {
	a0, a1, a2, a3 = a0[:len(b)], a1[:len(b)], a2[:len(b)], a3[:len(b)]
	var c0, c1, c2, c3 archsimd.Float32x16
	p := 0
	for ; p+avx512Lanes <= len(b); p += avx512Lanes {
		y := avx512LoadFull(b, p)
		c0 = avx512LoadFull(a0, p).MulAdd(y, c0)
		c1 = avx512LoadFull(a1, p).MulAdd(y, c1)
		c2 = avx512LoadFull(a2, p).MulAdd(y, c2)
		c3 = avx512LoadFull(a3, p).MulAdd(y, c3)
	}
	if p < len(b) {
		y := avx512LoadPart(b, p)
		c0 = avx512LoadPart(a0, p).MulAdd(y, c0)
		c1 = avx512LoadPart(a1, p).MulAdd(y, c1)
		c2 = avx512LoadPart(a2, p).MulAdd(y, c2)
		c3 = avx512LoadPart(a3, p).MulAdd(y, c3)
	}
	return avx512ReduceFour(c0, c1, c2, c3)
}

// avx512OneByOne returns the inner product of a and b. Reducing four copies
// of one accumulator gives lane 0 the same order every other path uses. a
// and b share a length.
func avx512OneByOne(a, b []float32) float32 {
	b = b[:len(a)]
	var c archsimd.Float32x16
	p := 0
	for ; p+avx512Lanes <= len(a); p += avx512Lanes {
		c = avx512LoadFull(a, p).MulAdd(avx512LoadFull(b, p), c)
	}
	if p < len(a) {
		c = avx512LoadPart(a, p).MulAdd(avx512LoadPart(b, p), c)
	}
	return avx512ReduceFour(c, c, c, c).GetElem(0)
}
