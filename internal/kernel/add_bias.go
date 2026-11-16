package kernel

// AddBias adds bias to every row of x in place, where x is a row-major
// [rows × cols] matrix and bias holds cols values. Callers use it after a
// linear projection that carries a bias term, and to apply an additive
// attention mask. The generic and SIMD builds give bitwise-identical results.
//
// It panics when len(x) != rows*cols or len(bias) != cols.
func AddBias(x, bias []float32, rows, cols int) {
	requireLength("x", len(x), rows*cols)
	requireLength("bias", len(bias), cols)
	addBias(x, bias, rows, cols)
}
