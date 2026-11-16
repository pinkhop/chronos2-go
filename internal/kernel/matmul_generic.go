package kernel

// matMulInnerUnroll is the number of rows of b folded into each pass over an
// output row, cutting loads and stores of the output by that factor.
const matMulInnerUnroll = 4

// MatMul computes out = a · b, where a is a row-major [m × k] matrix, b is a
// row-major [k × n] matrix, and out is a row-major [m × n] matrix. The
// network uses it to apply attention weights to values.
//
// Each output is accumulated in float32. out is fully overwritten and must
// not overlap a or b. It panics when a length disagrees with m, k, and n.
func MatMul(out, a, b []float32, m, k, n int) {
	clearUpperBits()
	requireLength("a", len(a), m*k)
	requireLength("b", len(b), k*n)
	requireLength("out", len(out), m*n)
	for row := range m {
		matMulRow(out[row*n:(row+1)*n], a[row*k:(row+1)*k], b, n)
	}
}

// matMulRow computes one output row as the sum of b's rows weighted by aRow,
// walking b row by row so every access is sequential.
func matMulRow(outRow, aRow, b []float32, n int) {
	clear(outRow)
	k := len(aRow)
	p := 0
	for ; p+matMulInnerUnroll <= k; p += matMulInnerUnroll {
		accumulateFourRows(outRow, aRow[p], aRow[p+1], aRow[p+2], aRow[p+3],
			b[p*n:(p+1)*n], b[(p+1)*n:(p+2)*n], b[(p+2)*n:(p+3)*n], b[(p+3)*n:(p+4)*n])
	}
	for ; p < k; p++ {
		accumulateRow(outRow, aRow[p], b[p*n:(p+1)*n])
	}
}

// accumulateFourRows adds w0·b0 + w1·b1 + w2·b2 + w3·b3 to outRow; every
// slice shares a length.
func accumulateFourRows(outRow []float32, w0, w1, w2, w3 float32, b0, b1, b2, b3 []float32) {
	b0 = b0[:len(outRow)]
	b1 = b1[:len(outRow)]
	b2 = b2[:len(outRow)]
	b3 = b3[:len(outRow)]
	for j := range outRow {
		outRow[j] += w0*b0[j] + w1*b1[j] + w2*b2[j] + w3*b3[j]
	}
}

// accumulateRow adds weight·bRow to outRow; both share a length.
func accumulateRow(outRow []float32, weight float32, bRow []float32) {
	bRow = bRow[:len(outRow)]
	for j := range outRow {
		outRow[j] += weight * bRow[j]
	}
}
