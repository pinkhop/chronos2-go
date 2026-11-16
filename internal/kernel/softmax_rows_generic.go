package kernel

import "math"

// softmaxRowsGeneric is the portable implementation of SoftmaxRows, compiled
// into every build; the SIMD build's tests compare against it. It
// exponentiates with math.Exp and sums each row strictly left to right. The
// length has already been checked by SoftmaxRows.
func softmaxRowsGeneric(x []float32, rows, cols int) {
	for r := range rows {
		softmaxRow(x[r*cols : (r+1)*cols])
	}
}

// softmaxRow replaces row with its softmax in place; see SoftmaxRows for the
// masking rules.
func softmaxRow(row []float32) {
	if len(row) == 0 {
		return
	}

	// Find the maximum score to shift by.
	maximum := row[0]
	for _, value := range row[1:] {
		maximum = max(maximum, value)
	}
	if math.IsInf(float64(maximum), -1) {
		uniform := 1 / float32(len(row))
		for i := range row {
			row[i] = uniform
		}
		return
	}

	// Exponentiate the shifted scores and normalize by their sum.
	var sum float32
	for i, value := range row {
		exponential := float32(math.Exp(float64(value - maximum)))
		row[i] = exponential
		sum += exponential
	}
	inverseSum := 1 / sum
	for i := range row {
		row[i] *= inverseSum
	}
}
