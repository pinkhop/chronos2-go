package kernel

// SoftmaxRows replaces each row of the row-major [rows × cols] matrix x with
// its softmax, in place, subtracting the row maximum first so large scores
// cannot overflow.
//
// Attention masks are additive: a masked score holds the most negative finite
// float32 and so receives zero weight. A row in which every score is masked
// becomes uniform, exactly as in PyTorch, because subtracting its maximum
// leaves every element at zero; each weight is then exactly 1/cols. A row of
// nothing but negative infinity also becomes uniform rather than NaN. A NaN
// score yields a NaN row. Each row is computed by one fixed procedure that
// depends only on that row, so results do not depend on rows or on how the
// caller splits them. The generic and SIMD builds compute the exponential and
// sum differently and agree within float32 rounding, not bitwise. It panics
// when len(x) != rows*cols.
func SoftmaxRows(x []float32, rows, cols int) {
	requireLength("x", len(x), rows*cols)
	softmaxRows(x, rows, cols)
}
