//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// avx512BlockRows is the number of rows of a that the AVX-512 register block
// covers. A 3×4 block's twelve accumulators, three chunks of a, one chunk of
// b, and the operand copies the compiler inserts fit in the 31 registers it
// allocates without spilling; a 4×4 block spills and measured about 25%
// slower.
const avx512BlockRows = 3

// avx512BlockColumns is the number of rows of b (output columns) the AVX-512
// register block covers; its four columns reduce into one 128-bit vector
// stored as four adjacent outputs.
const avx512BlockColumns = 4

// avx512ColumnTile is the number of rows of b processed together while every
// row of a streams past them; throughput varied by under 5% from 4 to 64
// rows.
const avx512ColumnTile = 8 * avx512BlockColumns

// avx512MatMulTransposed is MatMulTransposed using AVX-512 instructions.
// a ([m × k]), b ([n × k]), and out ([m × n]) are row-major with lengths
// already checked; out is fully overwritten and must not overlap a or b. The
// CPU must support AVX-512 and FMA.
//
// Every output, whether it falls in a 3×4 block, a leftover row, or a
// leftover column, is computed by one procedure: a Float32x16 accumulator
// gathers fused multiply-adds over k in chunks of sixteen, the final partial
// chunk is loaded zero-padded, and the sixteen lanes are reduced by
// avx512ReduceFour. An output therefore depends only on its row of a, its
// row of b, and k, never on m or on how the caller groups rows.
func avx512MatMulTransposed(out, a, b []float32, m, k, n int) {
	for columnStart := 0; columnStart < n; columnStart += avx512ColumnTile {
		columnEnd := min(columnStart+avx512ColumnTile, n)
		row := 0
		for ; row+avx512BlockRows <= m; row += avx512BlockRows {
			avx512ThreeRows(out, a, b, row, columnStart, columnEnd, k, n)
		}
		for ; row < m; row++ {
			avx512OneRow(out, a, b, row, columnStart, columnEnd, k, n)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// avx512ThreeRows fills out rows [row, row+3) across columns [columnStart,
// columnEnd), using the 3×4 micro-kernel where four columns remain and
// avx512FourByOne for the rest.
func avx512ThreeRows(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	a0 := a[row*k : (row+1)*k]
	a1 := a[(row+1)*k : (row+2)*k]
	a2 := a[(row+2)*k : (row+3)*k]
	column := columnStart
	for ; column+avx512BlockColumns <= columnEnd; column += avx512BlockColumns {
		r0, r1, r2 := avx512ThreeByFour(a0, a1, a2,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k])
		r0.StoreArray((*[avx2SumLanes]float32)(out[row*n+column:]))
		r1.StoreArray((*[avx2SumLanes]float32)(out[(row+1)*n+column:]))
		r2.StoreArray((*[avx2SumLanes]float32)(out[(row+2)*n+column:]))
	}
	for ; column < columnEnd; column++ {
		var sums [avx2SumLanes]float32
		avx512FourByOne(a0, a1, a2, a2, b[column*k:(column+1)*k]).StoreArray(&sums)
		for r := range avx512BlockRows {
			out[(row+r)*n+column] = sums[r]
		}
	}
}

// avx512OneRow fills out row row across columns [columnStart, columnEnd); it
// handles the rows left over when m is not a multiple of three, four columns
// at a time.
func avx512OneRow(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	aRow := a[row*k : (row+1)*k]
	outRow := out[row*n : (row+1)*n]
	column := columnStart
	for ; column+avx512BlockColumns <= columnEnd; column += avx512BlockColumns {
		avx512FourByOne(
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k], aRow,
		).StoreArray((*[avx2SumLanes]float32)(outRow[column:]))
	}
	for ; column < columnEnd; column++ {
		outRow[column] = avx512OneByOne(aRow, b[column*k:(column+1)*k])
	}
}
