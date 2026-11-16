package kernel

// addBiasReLUGeneric is the portable implementation of AddBiasReLU, compiled
// into every build; the SIMD builds' tests compare against it. Lengths have
// already been checked by AddBiasReLU.
func addBiasReLUGeneric(x, bias []float32, rows, cols int) {
	for r := range rows {
		addBiasReLURowGeneric(x[r*cols:(r+1)*cols], bias)
	}
}

// addBiasReLURowGeneric adds bias to row and clamps each sum at zero, element
// by element; the SIMD builds use it for the elements after their last full
// vector. row and bias share a length.
func addBiasReLURowGeneric(row, bias []float32) {
	row = row[:len(bias)]
	for c, value := range bias {
		// The built-in max returns NaN when either operand is NaN, as ReLU
		// does.
		row[c] = max(row[c]+value, 0)
	}
}
