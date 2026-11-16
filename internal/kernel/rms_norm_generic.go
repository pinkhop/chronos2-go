package kernel

import "math"

// RMSNorm writes the T5-style RMS normalization of each row of x into out:
// out = weight · x / sqrt(mean(x²) + eps). Unlike layer normalization, it
// subtracts no mean and adds no bias; every Chronos-2 norm uses this form.
//
// x and out are row-major [rows × cols] matrices and may alias; weight holds
// cols values; eps is the variance floor (1e-6 in Chronos-2) and must be
// non-negative. It panics when a length disagrees with rows and cols.
func RMSNorm(out, x, weight []float32, rows, cols int, eps float32) {
	clearUpperBits()
	requireLength("x", len(x), rows*cols)
	requireLength("out", len(out), rows*cols)
	requireLength("weight", len(weight), cols)
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		normalized := out[r*cols : (r+1)*cols]
		row = row[:len(weight)]
		normalized = normalized[:len(weight)]

		// Accumulate the mean square in float32, as the reference does.
		meanSquare := Dot(row, row) / float32(cols)
		inverseRMS := float32(1 / math.Sqrt(float64(meanSquare+eps)))

		// Scale before weighting to match the reference's operation order.
		for c, value := range row {
			normalized[c] = weight[c] * (value * inverseRMS)
		}
	}
}
