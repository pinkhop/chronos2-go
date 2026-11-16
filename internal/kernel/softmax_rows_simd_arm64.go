//go:build goexperiment.simd

package kernel

import (
	"math"
	"simd/archsimd"
)

// softmaxRows is SoftmaxRows after its length check, using NEON vector
// instructions on arm64. x is the row-major [rows × cols] matrix, replaced
// in place.
//
// Every row is computed by one procedure: a lane-wise maximum, then neonExp
// of each score minus that maximum with the exponentials summed in four
// lanes reduced as (l0+l1)+(l2+l3), then a multiply by 1/sum. A final
// partial chunk is padded with -∞, which leaves the maximum unchanged and
// exponentiates to exactly 0. The special rows match the generic build
// exactly: an all -∞ row gets exactly 1/cols, and a fully masked row
// exponentiates to exact ones whose sum is exactly cols, so each weight is
// exactly 1·(1/cols).
func softmaxRows(x []float32, rows, cols int) {
	for r := range rows {
		neonSoftmaxRow(x[r*cols : (r+1)*cols])
	}
}

// neonSoftmaxRow replaces row with its softmax in place; see SoftmaxRows for
// the masking rules.
func neonSoftmaxRow(row []float32) {
	if len(row) == 0 {
		return
	}
	maximum := neonRowMaximum(row)
	if math.IsInf(float64(maximum), -1) {
		uniform := 1 / float32(len(row))
		for i := range row {
			row[i] = uniform
		}
		return
	}
	inverseSum := 1 / neonExponentiateShifted(row, maximum)
	neonScaleRow(row, inverseSum)
}

// neonRowMaximum returns the largest value in row, or NaN when row holds a
// NaN, matching the built-in max the generic code uses. row is not empty.
func neonRowMaximum(row []float32) float32 {
	negativeInfinity := float32(math.Inf(-1))
	maxima := archsimd.BroadcastFloat32x4(negativeInfinity)
	p := 0
	for ; p+neonLanes <= len(row); p += neonLanes {
		maxima = maxima.Max(loadFull(row, p))
	}
	if p < len(row) {
		maxima = maxima.Max(loadPadded(row, p, negativeInfinity))
	}
	var lanes [neonLanes]float32
	maxima.StoreArray(&lanes)
	return max(lanes[0], lanes[1], lanes[2], lanes[3])
}

// neonExponentiateShifted replaces each value v of row with e^(v−maximum)
// and returns the sum of the results, accumulated in four lanes and reduced
// as (l0+l1)+(l2+l3). A final partial chunk runs through the same code
// padded with -∞, whose exponential is exactly 0.
func neonExponentiateShifted(row []float32, maximum float32) float32 {
	shift := archsimd.BroadcastFloat32x4(maximum)
	full := len(row) &^ (neonLanes - 1)
	sums := neonExponentiateChunks(row[:full], shift)
	if full < len(row) {
		negativeInfinity := float32(math.Inf(-1))
		tail := [neonLanes]float32{negativeInfinity, negativeInfinity, negativeInfinity, negativeInfinity}
		copy(tail[:], row[full:])
		sums = sums.Add(neonExponentiateChunks(tail[:], shift))
		copy(row[full:], tail[:])
	}
	return reduceFour(sums, sums, sums, sums).GetElem(0)
}

// neonScaleRow multiplies every value of row by scale.
func neonScaleRow(row []float32, scale float32) {
	factor := archsimd.BroadcastFloat32x4(scale)
	p := 0
	for ; p+neonLanes <= len(row); p += neonLanes {
		loadFull(row, p).Mul(factor).StoreArray((*[neonLanes]float32)(row[p : p+neonLanes]))
	}
	for ; p < len(row); p++ {
		row[p] *= scale
	}
}

// loadPadded returns the final partial chunk s[p:] as a vector with its
// missing lanes set to fill. p must be less than len(s).
func loadPadded(s []float32, p int, fill float32) archsimd.Float32x4 {
	chunk := [neonLanes]float32{fill, fill, fill, fill}
	copy(chunk[:], s[p:])
	return archsimd.LoadFloat32x4Array(&chunk)
}
