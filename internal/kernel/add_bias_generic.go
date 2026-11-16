package kernel

// addBiasGeneric is the portable implementation of AddBias, compiled into
// every build; the SIMD builds' tests compare against it. Lengths have
// already been checked by AddBias.
func addBiasGeneric(x, bias []float32, rows, cols int) {
	for r := range rows {
		addBiasRowGeneric(x[r*cols:(r+1)*cols], bias)
	}
}

// addBiasRowGeneric adds bias to row element by element; the SIMD builds use
// it for the elements after their last full vector. row and bias share a
// length.
func addBiasRowGeneric(row, bias []float32) {
	row = row[:len(bias)]
	for c, value := range bias {
		row[c] += value
	}
}
