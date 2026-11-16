package kernel

// AddBiasReLU adds bias to every row of x and then replaces each negative
// sum with zero, in place, where x is a row-major [rows × cols] matrix and
// bias holds cols values. It gives exactly the result of AddBias followed by
// ReLU in a single pass over x, for a projection that feeds a ReLU; NaN
// elements stay NaN. The generic and SIMD builds give bitwise-identical
// results apart from the sign and payload of NaN.
//
// It panics when len(x) != rows*cols or len(bias) != cols.
func AddBiasReLU(x, bias []float32, rows, cols int) {
	requireLength("x", len(x), rows*cols)
	requireLength("bias", len(bias), cols)
	addBiasReLU(x, bias, rows, cols)
}
