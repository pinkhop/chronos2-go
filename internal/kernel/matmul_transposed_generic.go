package kernel

// transposedColumnTile is the number of rows of b (output columns) processed
// together. At the model widths, 32 rows of 768 or 512 float32 values (96 or
// 64 KiB) stay resident in the L1 data cache of current desktop and mobile
// cores while every row of a streams past them; measured throughput falls off
// on either side of this size.
const transposedColumnTile = 32

// transposedBlock is the edge of the square register block: each micro-kernel
// call computes a transposedBlock × transposedBlock tile of out.
const transposedBlock = 4

// matMulTransposedGeneric is the portable implementation of
// MatMulTransposed, compiled into every build. The SIMD builds use it where the
// CPU lacks the vector features they need, and their tests compare against it.
// Each output is accumulated in float32 strictly in order of k. Lengths have
// already been checked by MatMulTransposed.
func matMulTransposedGeneric(out, a, b []float32, m, k, n int) {
	for columnStart := 0; columnStart < n; columnStart += transposedColumnTile {
		columnEnd := min(columnStart+transposedColumnTile, n)
		row := 0
		for ; row+transposedBlock <= m; row += transposedBlock {
			transposedFourRows(out, a, b, row, columnStart, columnEnd, k, n)
		}
		for ; row < m; row++ {
			transposedOneRow(out, a, b, row, columnStart, columnEnd, k, n)
		}
	}
}

// transposedFourRows fills out rows [row, row+4) across columns
// [columnStart, columnEnd), using the 4×4 micro-kernel where four columns
// remain and dotFourByOne for the rest.
func transposedFourRows(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	a0 := a[row*k : (row+1)*k]
	a1 := a[(row+1)*k : (row+2)*k]
	a2 := a[(row+2)*k : (row+3)*k]
	a3 := a[(row+3)*k : (row+4)*k]
	column := columnStart
	for ; column+transposedBlock <= columnEnd; column += transposedBlock {
		tile := dotFourByFour(a0, a1, a2, a3,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k])
		for r := range transposedBlock {
			copy(out[(row+r)*n+column:(row+r)*n+column+transposedBlock], tile[r*transposedBlock:(r+1)*transposedBlock])
		}
	}
	for ; column < columnEnd; column++ {
		s0, s1, s2, s3 := dotFourByOne(a0, a1, a2, a3, b[column*k:(column+1)*k])
		out[row*n+column] = s0
		out[(row+1)*n+column] = s1
		out[(row+2)*n+column] = s2
		out[(row+3)*n+column] = s3
	}
}

// transposedOneRow fills out row row across columns [columnStart,
// columnEnd); it handles the rows left over when m is not a multiple of four,
// four columns at a time so independent sums keep the arithmetic units busy.
func transposedOneRow(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	aRow := a[row*k : (row+1)*k]
	outRow := out[row*n : (row+1)*n]
	column := columnStart
	for ; column+transposedBlock <= columnEnd; column += transposedBlock {
		outRow[column], outRow[column+1], outRow[column+2], outRow[column+3] = dotOneByFour(aRow,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k])
	}
	for ; column < columnEnd; column++ {
		outRow[column] = sequentialDot(aRow, b[column*k:(column+1)*k])
	}
}

// dotOneByFour returns the inner products of a with b0, b1, b2, and b3, each
// summed strictly left to right. All slices share a length.
func dotOneByFour(a, b0, b1, b2, b3 []float32) (s0, s1, s2, s3 float32) {
	b0 = b0[:len(a)]
	b1 = b1[:len(a)]
	b2 = b2[:len(a)]
	b3 = b3[:len(a)]
	for p, x := range a {
		s0 += x * b0[p]
		s1 += x * b1[p]
		s2 += x * b2[p]
		s3 += x * b3[p]
	}
	return s0, s1, s2, s3
}

// dotFourByOne returns the inner products of a0, a1, a2, and a3 with b, each
// summed strictly left to right. All slices share a length.
func dotFourByOne(a0, a1, a2, a3, b []float32) (s0, s1, s2, s3 float32) {
	a0 = a0[:len(b)]
	a1 = a1[:len(b)]
	a2 = a2[:len(b)]
	a3 = a3[:len(b)]
	for p, y := range b {
		s0 += a0[p] * y
		s1 += a1[p] * y
		s2 += a2[p] * y
		s3 += a3[p] * y
	}
	return s0, s1, s2, s3
}

// sequentialDot returns the inner product of a and b summed strictly left to
// right, the order every micro-kernel here uses for each output. Sharing one
// order inside and outside whole 4×4 blocks makes every output independent of
// how the caller splits m, so a forward pass gives identical results at any
// parallelism. a and b share a length.
func sequentialDot(a, b []float32) float32 {
	b = b[:len(a)]
	var sum float32
	for p, x := range a {
		sum += x * b[p]
	}
	return sum
}

// dotFourByFour returns the sixteen inner products of a0 through a3 with b0
// through b3 as a row-major 4×4 tile (element 4·R + C pairs aR with bC).
// Computing them together loads each element once for four products, which
// keeps the arithmetic units busy instead of waiting on memory. All slices
// share a length.
func dotFourByFour(a0, a1, a2, a3, b0, b1, b2, b3 []float32) [transposedBlock * transposedBlock]float32 {
	// Reslicing to a common length lets the compiler drop bounds checks.
	a1 = a1[:len(a0)]
	a2 = a2[:len(a0)]
	a3 = a3[:len(a0)]
	b0 = b0[:len(a0)]
	b1 = b1[:len(a0)]
	b2 = b2[:len(a0)]
	b3 = b3[:len(a0)]

	// Sixteen named accumulators stay in registers; an array would not.
	var s00, s01, s02, s03, s10, s11, s12, s13 float32
	var s20, s21, s22, s23, s30, s31, s32, s33 float32
	for p, x0 := range a0 {
		x1, x2, x3 := a1[p], a2[p], a3[p]
		y0, y1, y2, y3 := b0[p], b1[p], b2[p], b3[p]
		s00 += x0 * y0
		s01 += x0 * y1
		s02 += x0 * y2
		s03 += x0 * y3
		s10 += x1 * y0
		s11 += x1 * y1
		s12 += x1 * y2
		s13 += x1 * y3
		s20 += x2 * y0
		s21 += x2 * y1
		s22 += x2 * y2
		s23 += x2 * y3
		s30 += x3 * y0
		s31 += x3 * y1
		s32 += x3 * y2
		s33 += x3 * y3
	}
	return [transposedBlock * transposedBlock]float32{
		s00, s01, s02, s03, s10, s11, s12, s13,
		s20, s21, s22, s23, s30, s31, s32, s33,
	}
}
