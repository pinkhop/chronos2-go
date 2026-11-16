//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// neonFourByFive returns the twenty inner products of a0 through a3 with b0
// through b4. Result R holds aR with b0 through b3 in lanes 0 through 3, ready
// to store as four adjacent outputs; lastColumn holds a0 through a3 with b4,
// ready to store down one output column. The full-chunk branch is taken on
// every iteration but the last, so it costs one predictable compare per
// chunk instead of one per load. All slices share a length.
func neonFourByFive(a0, a1, a2, a3, b0, b1, b2, b3, b4 []float32) (r0, r1, r2, r3, lastColumn archsimd.Float32x4) {
	// Reslicing to a common length lets the compiler drop bounds checks.
	a1, a2, a3 = a1[:len(a0)], a2[:len(a0)], a3[:len(a0)]
	b0, b1, b2, b3, b4 = b0[:len(a0)], b1[:len(a0)], b2[:len(a0)], b3[:len(a0)], b4[:len(a0)]

	var c00, c01, c02, c03, c04, c10, c11, c12, c13, c14 archsimd.Float32x4
	var c20, c21, c22, c23, c24, c30, c31, c32, c33, c34 archsimd.Float32x4
	for p := 0; p < len(a0); p += neonLanes {
		var x0, x1, x2, x3, y0, y1, y2, y3, y4 archsimd.Float32x4
		if p+neonLanes <= len(a0) {
			x0, x1, x2, x3 = loadFull(a0, p), loadFull(a1, p), loadFull(a2, p), loadFull(a3, p)
			y0, y1, y2, y3, y4 = loadFull(b0, p), loadFull(b1, p), loadFull(b2, p), loadFull(b3, p), loadFull(b4, p)
		} else {
			x0, x1, x2, x3 = loadPart(a0, p), loadPart(a1, p), loadPart(a2, p), loadPart(a3, p)
			y0, y1, y2, y3, y4 = loadPart(b0, p), loadPart(b1, p), loadPart(b2, p), loadPart(b3, p), loadPart(b4, p)
		}
		c00, c01, c02, c03, c04 = x0.MulAdd(y0, c00), x0.MulAdd(y1, c01), x0.MulAdd(y2, c02), x0.MulAdd(y3, c03), x0.MulAdd(y4, c04)
		c10, c11, c12, c13, c14 = x1.MulAdd(y0, c10), x1.MulAdd(y1, c11), x1.MulAdd(y2, c12), x1.MulAdd(y3, c13), x1.MulAdd(y4, c14)
		c20, c21, c22, c23, c24 = x2.MulAdd(y0, c20), x2.MulAdd(y1, c21), x2.MulAdd(y2, c22), x2.MulAdd(y3, c23), x2.MulAdd(y4, c24)
		c30, c31, c32, c33, c34 = x3.MulAdd(y0, c30), x3.MulAdd(y1, c31), x3.MulAdd(y2, c32), x3.MulAdd(y3, c33), x3.MulAdd(y4, c34)
	}
	return reduceFour(c00, c01, c02, c03), reduceFour(c10, c11, c12, c13),
		reduceFour(c20, c21, c22, c23), reduceFour(c30, c31, c32, c33),
		reduceFour(c04, c14, c24, c34)
}

// neonFourByOne returns the inner products of a0, a1, a2, and a3 with b in
// lanes 0 through 3. All slices share a length.
func neonFourByOne(a0, a1, a2, a3, b []float32) archsimd.Float32x4 {
	a0 = a0[:len(b)]
	a1 = a1[:len(b)]
	a2 = a2[:len(b)]
	a3 = a3[:len(b)]
	var c0, c1, c2, c3 archsimd.Float32x4
	for p := 0; p < len(b); p += neonLanes {
		var x0, x1, x2, x3, y archsimd.Float32x4
		if p+neonLanes <= len(b) {
			x0, x1, x2, x3, y = loadFull(a0, p), loadFull(a1, p), loadFull(a2, p), loadFull(a3, p), loadFull(b, p)
		} else {
			x0, x1, x2, x3, y = loadPart(a0, p), loadPart(a1, p), loadPart(a2, p), loadPart(a3, p), loadPart(b, p)
		}
		c0 = x0.MulAdd(y, c0)
		c1 = x1.MulAdd(y, c1)
		c2 = x2.MulAdd(y, c2)
		c3 = x3.MulAdd(y, c3)
	}
	return reduceFour(c0, c1, c2, c3)
}

// neonOneByFour returns the inner products of a with b0, b1, b2, and b3 in
// lanes 0 through 3. All slices share a length.
func neonOneByFour(a, b0, b1, b2, b3 []float32) archsimd.Float32x4 {
	b0 = b0[:len(a)]
	b1 = b1[:len(a)]
	b2 = b2[:len(a)]
	b3 = b3[:len(a)]
	var c0, c1, c2, c3 archsimd.Float32x4
	for p := 0; p < len(a); p += neonLanes {
		var x, y0, y1, y2, y3 archsimd.Float32x4
		if p+neonLanes <= len(a) {
			x, y0, y1, y2, y3 = loadFull(a, p), loadFull(b0, p), loadFull(b1, p), loadFull(b2, p), loadFull(b3, p)
		} else {
			x, y0, y1, y2, y3 = loadPart(a, p), loadPart(b0, p), loadPart(b1, p), loadPart(b2, p), loadPart(b3, p)
		}
		c0 = x.MulAdd(y0, c0)
		c1 = x.MulAdd(y1, c1)
		c2 = x.MulAdd(y2, c2)
		c3 = x.MulAdd(y3, c3)
	}
	return reduceFour(c0, c1, c2, c3)
}

// neonOneByOne returns the inner product of a and b. Reducing four copies of
// one accumulator gives lane 0 the same (l0+l1)+(l2+l3) order every other path
// uses. a and b share a length.
func neonOneByOne(a, b []float32) float32 {
	b = b[:len(a)]
	var c archsimd.Float32x4
	for p := 0; p < len(a); p += neonLanes {
		var x, y archsimd.Float32x4
		if p+neonLanes <= len(a) {
			x, y = loadFull(a, p), loadFull(b, p)
		} else {
			x, y = loadPart(a, p), loadPart(b, p)
		}
		c = x.MulAdd(y, c)
	}
	return reduceFour(c, c, c, c).GetElem(0)
}
