//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx2BlockRows is the number of rows of a that the AVX2 register block
// covers. A 2×4 block keeps eight accumulators, enough independent fused
// multiply-adds to cover their latency on both FMA units; it measured about
// 12% faster than a 3×4 block, whose twelve accumulators spill.
const avx2BlockRows = 2

// avx2BlockColumns is the number of rows of b (output columns) the AVX2
// register block covers; its four columns reduce into one 128-bit vector
// stored as four adjacent outputs.
const avx2BlockColumns = 4

// avx2ColumnTile is the number of rows of b processed together while every
// row of a streams past them.
const avx2ColumnTile = 16 * avx2BlockColumns

// avx2MatMulTransposed is MatMulTransposed using AVX2 and FMA instructions.
// a ([m × k]), b ([n × k]), and out ([m × n]) are row-major with lengths
// already checked; out is fully overwritten and must not overlap a or b. The
// CPU must support AVX2 and FMA.
//
// Every output, whether it falls in a 2×4 block, a leftover row, or a
// leftover column, is computed by one procedure: a Float32x8 accumulator
// gathers fused multiply-adds over k in chunks of eight, the final partial
// chunk is loaded zero-padded so it adds exact zeros, and the eight lanes are
// reduced as ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)). An output therefore
// depends only on its row of a, its row of b, and k, never on m or on how the
// caller groups rows. It clears the upper vector-register bits before
// returning (see clearUpperBits).
func avx2MatMulTransposed(out, a, b []float32, m, k, n int) {
	for columnStart := 0; columnStart < n; columnStart += avx2ColumnTile {
		columnEnd := min(columnStart+avx2ColumnTile, n)
		row := 0
		for ; row+avx2BlockRows <= m; row += avx2BlockRows {
			avx2TwoRows(out, a, b, row, columnStart, columnEnd, k, n)
		}
		if row < m {
			avx2OneRow(out, a, b, row, columnStart, columnEnd, k, n)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// avx2TwoRows fills out rows row and row+1 across columns [columnStart,
// columnEnd), using the 2×4 micro-kernel where four columns remain and
// avx2TwoByOne for the rest.
func avx2TwoRows(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	a0 := a[row*k : (row+1)*k]
	a1 := a[(row+1)*k : (row+2)*k]
	column := columnStart
	for ; column+avx2BlockColumns <= columnEnd; column += avx2BlockColumns {
		r0, r1 := avx2TwoByFour(a0, a1,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k])
		r0.StoreArray((*[avx2SumLanes]float32)(out[row*n+column:]))
		r1.StoreArray((*[avx2SumLanes]float32)(out[(row+1)*n+column:]))
	}
	for ; column < columnEnd; column++ {
		var sums [avx2SumLanes]float32
		avx2TwoByOne(a0, a1, b[column*k:(column+1)*k]).StoreArray(&sums)
		out[row*n+column] = sums[0]
		out[(row+1)*n+column] = sums[1]
	}
}

// avx2OneRow fills out row row across columns [columnStart, columnEnd); it
// handles the row left over when m is odd, four columns at a time.
func avx2OneRow(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	aRow := a[row*k : (row+1)*k]
	outRow := out[row*n : (row+1)*n]
	column := columnStart
	for ; column+avx2BlockColumns <= columnEnd; column += avx2BlockColumns {
		avx2OneByFour(aRow,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k],
		).StoreArray((*[avx2SumLanes]float32)(outRow[column:]))
	}
	for ; column < columnEnd; column++ {
		outRow[column] = avx2OneByOne(aRow, b[column*k:(column+1)*k])
	}
}
