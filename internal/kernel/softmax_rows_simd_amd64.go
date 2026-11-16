//go:build goexperiment.simd

package kernel

import (
	"math"
	"simd/archsimd"
)

// softmaxRows is SoftmaxRows after its length check. It runs the widest
// implementation the CPU supports, chosen once per process.
func softmaxRows(x []float32, rows, cols int) {
	switch detectedVectorLevel {
	case vectorLevelAVX2, vectorLevelAVX512:
		avx2SoftmaxRows(x, rows, cols)
	case vectorLevelUnspecified, vectorLevelGeneric:
		softmaxRowsGeneric(x, rows, cols)
	}
}

// avx2SoftmaxRows is SoftmaxRows using AVX2 and FMA instructions. x is the
// row-major [rows × cols] matrix, replaced in place. The CPU must support
// AVX2 and FMA.
//
// Every row is computed by one procedure: a lane-wise maximum, then
// avx2ExponentiateChunks of each score minus that maximum with the
// exponentials summed in eight lanes reduced as
// ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), then a multiply by 1/sum. A final
// partial chunk is padded with -∞, which leaves the maximum unchanged and
// exponentiates to exactly 0. The special rows match the generic build
// exactly: an all -∞ row gets exactly 1/cols, and a fully masked row
// exponentiates to exact ones whose sum is exactly cols, so each weight is
// exactly 1·(1/cols). It clears the upper vector-register bits before
// returning (see clearUpperBits).
func avx2SoftmaxRows(x []float32, rows, cols int) {
	for r := range rows {
		avx2SoftmaxRow(x[r*cols : (r+1)*cols])
	}
	archsimd.ClearAVXUpperBits()
}

// avx2SoftmaxRow replaces row with its softmax in place; see SoftmaxRows for
// the masking rules.
func avx2SoftmaxRow(row []float32) {
	if len(row) == 0 {
		return
	}
	maximum := avx2RowMaximum(row)
	if math.IsInf(float64(maximum), -1) {
		uniform := 1 / float32(len(row))
		for i := range row {
			row[i] = uniform
		}
		return
	}
	inverseSum := 1 / avx2ExponentiateShifted(row, maximum)
	avx2ScaleRow(row, inverseSum)
}

// avx2RowMaximum returns the largest value in row, or NaN when row holds a
// NaN, matching the built-in max the generic code uses. x86's maximum
// instruction does not propagate NaN, so NaN lanes are tracked separately.
// row is not empty.
func avx2RowMaximum(row []float32) float32 {
	negativeInfinity := float32(math.Inf(-1))
	maxima := archsimd.BroadcastFloat32x8(negativeInfinity)
	nanLanes := maxima.IsNaN()
	p := 0
	for ; p+avx2Lanes <= len(row); p += avx2Lanes {
		chunk := avx2LoadFull(row, p)
		maxima = maxima.Max(chunk)
		nanLanes = nanLanes.Or(chunk.IsNaN())
	}
	if p < len(row) {
		chunk := avx2LoadPadded(row, p, negativeInfinity)
		maxima = maxima.Max(chunk)
		nanLanes = nanLanes.Or(chunk.IsNaN())
	}
	if nanLanes.ToBits() != 0 {
		return float32(math.NaN())
	}
	var lanes [avx2Lanes]float32
	maxima.StoreArray(&lanes)
	return max(lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7])
}

// avx2ExponentiateShifted replaces each value v of row with e^(v−maximum)
// and returns the sum of the results, accumulated in eight lanes and reduced
// as ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)). A final partial chunk runs
// through the same code padded with -∞, whose exponential is exactly 0.
func avx2ExponentiateShifted(row []float32, maximum float32) float32 {
	shift := archsimd.BroadcastFloat32x8(maximum)
	full := len(row) &^ (avx2Lanes - 1)
	sums := avx2ExponentiateChunks(row[:full], shift)
	if full < len(row) {
		negativeInfinity := float32(math.Inf(-1))
		tail := [avx2Lanes]float32{
			negativeInfinity, negativeInfinity, negativeInfinity, negativeInfinity,
			negativeInfinity, negativeInfinity, negativeInfinity, negativeInfinity,
		}
		copy(tail[:], row[full:])
		sums = sums.Add(avx2ExponentiateChunks(tail[:], shift))
		copy(row[full:], tail[:])
	}
	return avx2ReduceFour(sums, sums, sums, sums).GetElem(0)
}

// avx2ScaleRow multiplies every value of row by scale.
func avx2ScaleRow(row []float32, scale float32) {
	factor := archsimd.BroadcastFloat32x8(scale)
	p := 0
	for ; p+avx2Lanes <= len(row); p += avx2Lanes {
		avx2LoadFull(row, p).Mul(factor).StoreArray((*[avx2Lanes]float32)(row[p : p+avx2Lanes]))
	}
	for ; p < len(row); p++ {
		row[p] *= scale
	}
}
