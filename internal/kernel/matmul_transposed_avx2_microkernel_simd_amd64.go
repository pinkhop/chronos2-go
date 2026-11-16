//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// The compiler implements MulAdd with the VFMADD213PS form, which writes the
// result over the receiver's register. Each fused multiply-add below
// therefore makes the receiver an operand at its last use where one exists,
// so the compiler need not copy a still-needed operand first; this measured
// about 10% faster than always making a row of a the receiver.

// avx2TwoByFour returns the eight inner products of a0 and a1 with b0
// through b3. Result R holds aR with b0 through b3 in lanes 0 through 3, ready
// to store as four adjacent outputs. Each chunk of b is loaded once and used
// at once by both rows, so the loop holds eight accumulators and three loaded
// chunks, which leaves AVX2's sixteen vector registers room to avoid
// spilling. All slices share a length.
func avx2TwoByFour(a0, a1, b0, b1, b2, b3 []float32) (r0, r1 archsimd.Float32x4) {
	// Reslicing to a common length lets the compiler drop bounds checks.
	a1 = a1[:len(a0)]
	b0, b1, b2, b3 = b0[:len(a0)], b1[:len(a0)], b2[:len(a0)], b3[:len(a0)]

	var c00, c01, c02, c03, c10, c11, c12, c13 archsimd.Float32x8
	p := 0
	for ; p+avx2Lanes <= len(a0); p += avx2Lanes {
		x0, x1 := avx2LoadFull(a0, p), avx2LoadFull(a1, p)
		y := avx2LoadFull(b0, p)
		c00, c10 = x0.MulAdd(y, c00), y.MulAdd(x1, c10)
		y = avx2LoadFull(b1, p)
		c01, c11 = x0.MulAdd(y, c01), y.MulAdd(x1, c11)
		y = avx2LoadFull(b2, p)
		c02, c12 = x0.MulAdd(y, c02), y.MulAdd(x1, c12)
		y = avx2LoadFull(b3, p)
		c03, c13 = x0.MulAdd(y, c03), y.MulAdd(x1, c13)
	}
	if p < len(a0) {
		x0, x1 := avx2LoadPart(a0, p), avx2LoadPart(a1, p)
		y := avx2LoadPart(b0, p)
		c00, c10 = x0.MulAdd(y, c00), y.MulAdd(x1, c10)
		y = avx2LoadPart(b1, p)
		c01, c11 = x0.MulAdd(y, c01), y.MulAdd(x1, c11)
		y = avx2LoadPart(b2, p)
		c02, c12 = x0.MulAdd(y, c02), y.MulAdd(x1, c12)
		y = avx2LoadPart(b3, p)
		c03, c13 = x0.MulAdd(y, c03), y.MulAdd(x1, c13)
	}
	return avx2ReduceFour(c00, c01, c02, c03), avx2ReduceFour(c10, c11, c12, c13)
}

// avx2TwoByOne returns the inner products of a0 and a1 with b in lanes 0 and
// 1; lanes 2 and 3 repeat them. All slices share a length.
func avx2TwoByOne(a0, a1, b []float32) archsimd.Float32x4 {
	a0, a1 = a0[:len(b)], a1[:len(b)]
	var c0, c1 archsimd.Float32x8
	p := 0
	for ; p+avx2Lanes <= len(b); p += avx2Lanes {
		y := avx2LoadFull(b, p)
		c0 = avx2LoadFull(a0, p).MulAdd(y, c0)
		c1 = avx2LoadFull(a1, p).MulAdd(y, c1)
	}
	if p < len(b) {
		y := avx2LoadPart(b, p)
		c0 = avx2LoadPart(a0, p).MulAdd(y, c0)
		c1 = avx2LoadPart(a1, p).MulAdd(y, c1)
	}
	return avx2ReduceFour(c0, c1, c0, c1)
}

// avx2OneByFour returns the inner products of a with b0, b1, b2, and b3 in
// lanes 0 through 3. All slices share a length.
func avx2OneByFour(a, b0, b1, b2, b3 []float32) archsimd.Float32x4 {
	b0, b1, b2, b3 = b0[:len(a)], b1[:len(a)], b2[:len(a)], b3[:len(a)]
	var c0, c1, c2, c3 archsimd.Float32x8
	p := 0
	for ; p+avx2Lanes <= len(a); p += avx2Lanes {
		x := avx2LoadFull(a, p)
		c0 = avx2LoadFull(b0, p).MulAdd(x, c0)
		c1 = avx2LoadFull(b1, p).MulAdd(x, c1)
		c2 = avx2LoadFull(b2, p).MulAdd(x, c2)
		c3 = avx2LoadFull(b3, p).MulAdd(x, c3)
	}
	if p < len(a) {
		x := avx2LoadPart(a, p)
		c0 = avx2LoadPart(b0, p).MulAdd(x, c0)
		c1 = avx2LoadPart(b1, p).MulAdd(x, c1)
		c2 = avx2LoadPart(b2, p).MulAdd(x, c2)
		c3 = avx2LoadPart(b3, p).MulAdd(x, c3)
	}
	return avx2ReduceFour(c0, c1, c2, c3)
}

// avx2OneByOne returns the inner product of a and b. Reducing four copies of
// one accumulator gives lane 0 the same order every other path uses. a and b
// share a length.
func avx2OneByOne(a, b []float32) float32 {
	b = b[:len(a)]
	var c archsimd.Float32x8
	p := 0
	for ; p+avx2Lanes <= len(a); p += avx2Lanes {
		c = avx2LoadFull(a, p).MulAdd(avx2LoadFull(b, p), c)
	}
	if p < len(a) {
		c = avx2LoadPart(a, p).MulAdd(avx2LoadPart(b, p), c)
	}
	return avx2ReduceFour(c, c, c, c).GetElem(0)
}
