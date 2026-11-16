//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx2Exp returns eˣ in each lane of x by running avx2ExponentiateChunks on
// one chunk; it exists so tests can evaluate the softmax exponential
// directly. The CPU must support AVX2 and FMA.
func avx2Exp(x archsimd.Float32x8) archsimd.Float32x8 {
	var chunk [avx2Lanes]float32
	x.StoreArray(&chunk)
	var zero archsimd.Float32x8
	avx2ExponentiateChunks(chunk[:], zero)
	return archsimd.LoadFloat32x8Array(&chunk)
}

// avx2ExponentiateChunks replaces each value v of values with e^(v−shift)
// and returns the lane-wise sums of the results; len(values) must be a
// multiple of eight. The exponential is valid for v − shift ≤ 0, the range
// softmax produces: it is within 2 ulp of the correctly rounded result,
// including subnormal results, returns exactly 1 for ±0, exactly 0 for every
// argument below about -103.97 (masked scores and -∞ among them), and NaN for
// NaN. Constants are broadcast once per call so the loop holds only
// arithmetic. The CPU must support AVX2 and FMA.
func avx2ExponentiateChunks(values []float32, shift archsimd.Float32x8) archsimd.Float32x8 {
	clamp, log2E := archsimd.BroadcastFloat32x8(expClamp), archsimd.BroadcastFloat32x8(expLog2E)
	ln2High, ln2Low := archsimd.BroadcastFloat32x8(expLn2High), archsimd.BroadcastFloat32x8(expLn2Low)
	c0, c1 := archsimd.BroadcastFloat32x8(expCoefficient0), archsimd.BroadcastFloat32x8(expCoefficient1)
	c2, c3 := archsimd.BroadcastFloat32x8(expCoefficient2), archsimd.BroadcastFloat32x8(expCoefficient3)
	c4, c5 := archsimd.BroadcastFloat32x8(expCoefficient4), archsimd.BroadcastFloat32x8(expCoefficient5)
	one, underflow := archsimd.BroadcastFloat32x8(1), archsimd.BroadcastFloat32x8(expUnderflow)
	bias := archsimd.BroadcastInt32x8(expExponentBias)
	var zero, sums archsimd.Float32x8
	for p := 0; p+avx2Lanes <= len(values); p += avx2Lanes {
		x := avx2LoadFull(values, p).Sub(shift)

		// Reduce: x = n·ln 2 + r. A NaN lane computes garbage here and is
		// restored below, because x86's maximum does not propagate NaN.
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
		scaled := er.Mul(avx2PowerOfTwo(half, bias)).Mul(avx2PowerOfTwo(exponent.Sub(half), bias))

		// Zero the lanes that underflow, then pass NaN lanes through.
		result := zero.IfElse(x.Less(underflow), scaled)
		result = x.IfElse(x.IsNaN(), result)
		result.StoreArray((*[avx2Lanes]float32)(values[p : p+avx2Lanes]))
		sums = sums.Add(result)
	}
	return sums
}

// avx2PowerOfTwo returns 2ᵉ in each lane by adding bias (a broadcast 127) to
// e and writing it into the float32 exponent field; e must lie in
// [-126, 127].
func avx2PowerOfTwo(e, bias archsimd.Int32x8) archsimd.Float32x8 {
	return e.Add(bias).ShiftAllLeft(expMantissaBits).ToBits().BitsToFloat32()
}
