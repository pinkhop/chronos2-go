//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// neonExp returns eˣ in each lane of x by running neonExponentiateChunks on
// one chunk; it exists so tests can evaluate the softmax exponential
// directly.
func neonExp(x archsimd.Float32x4) archsimd.Float32x4 {
	var chunk [neonLanes]float32
	x.StoreArray(&chunk)
	var zero archsimd.Float32x4
	neonExponentiateChunks(chunk[:], zero)
	return archsimd.LoadFloat32x4Array(&chunk)
}

// neonExponentiateChunks replaces each value v of values with e^(v−shift)
// and returns the lane-wise sums of the results; len(values) must be a
// multiple of four. The exponential is valid for v − shift ≤ 0, the range
// softmax produces: it is within 2 ulp of the correctly rounded result,
// including subnormal results, returns exactly 1 for ±0, exactly 0 for
// every argument below about -103.97 (masked scores and -∞ among them), and
// NaN for NaN. Constants are broadcast once per call so the loop holds only
// arithmetic.
func neonExponentiateChunks(values []float32, shift archsimd.Float32x4) archsimd.Float32x4 {
	clamp, log2E := archsimd.BroadcastFloat32x4(expClamp), archsimd.BroadcastFloat32x4(expLog2E)
	ln2High, ln2Low := archsimd.BroadcastFloat32x4(expLn2High), archsimd.BroadcastFloat32x4(expLn2Low)
	c0, c1 := archsimd.BroadcastFloat32x4(expCoefficient0), archsimd.BroadcastFloat32x4(expCoefficient1)
	c2, c3 := archsimd.BroadcastFloat32x4(expCoefficient2), archsimd.BroadcastFloat32x4(expCoefficient3)
	c4, c5 := archsimd.BroadcastFloat32x4(expCoefficient4), archsimd.BroadcastFloat32x4(expCoefficient5)
	one, underflow := archsimd.BroadcastFloat32x4(1), archsimd.BroadcastFloat32x4(expUnderflow)
	bias := archsimd.BroadcastInt32x4(expExponentBias)
	var zero, sums archsimd.Float32x4
	for p := 0; p+neonLanes <= len(values); p += neonLanes {
		x := loadFull(values, p).Sub(shift)

		// Reduce: x = n·ln 2 + r. Max propagates NaN, so NaN stays NaN.
		clamped := x.Max(clamp)
		n := clamped.Mul(log2E).Round()
		negativeN := n.Neg()
		r := negativeN.MulAdd(ln2Low, negativeN.MulAdd(ln2High, clamped))

		// Approximate eʳ = 1 + r + r²·P(r); at r = 0 this is exactly 1.
		polynomial := c5.MulAdd(r, c4).MulAdd(r, c3).MulAdd(r, c2).MulAdd(r, c1).MulAdd(r, c0)
		er := polynomial.MulAdd(r.Mul(r), r).Add(one)

		// Scale by 2ⁿ as 2^⌊n/2⌋ · 2^(n−⌊n/2⌋): both factors stay normal
		// down to n = -150, the first product is exact, and only the second
		// rounds, so subnormal results are correctly scaled.
		exponent := n.ConvertToInt32()
		half := exponent.ShiftAllRight(1)
		scaled := er.Mul(powerOfTwo(half, bias)).Mul(powerOfTwo(exponent.Sub(half), bias))

		// Zero the lanes that underflow; NaN compares false and is kept.
		result := zero.IfElse(x.Less(underflow), scaled)
		result.StoreArray((*[neonLanes]float32)(values[p : p+neonLanes]))
		sums = sums.Add(result)
	}
	return sums
}

// powerOfTwo returns 2ᵉ in each lane by adding bias (a broadcast 127) to e
// and writing it into the float32 exponent field; e must lie in [-126, 127].
func powerOfTwo(e, bias archsimd.Int32x4) archsimd.Float32x4 {
	return e.Add(bias).ShiftAllLeft(expMantissaBits).ToBits().BitsToFloat32()
}
