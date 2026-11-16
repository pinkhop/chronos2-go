//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// neonLanes is the number of float32 lanes in a NEON vector register.
const neonLanes = 4

// neonBlockRows is the number of rows of a that the NEON register block
// covers; it equals neonLanes so one vector carries the block's four rows of
// a single output column.
const neonBlockRows = 4

// neonBlockColumns is the number of rows of b (output columns) the NEON
// register block covers. A 4×5 block keeps twenty accumulators and nine loaded
// chunks in 29 of NEON's 32 vector registers, the widest block that does not
// spill; it measured about 15% faster than 4×4, whose loop spends a larger
// share of its instructions on addressing.
const neonBlockColumns = 5

// neonColumnTile is the number of rows of b processed together: eight 4×5
// blocks. At k = 768, the tile's 40 rows (120 KiB) stay in the L1 data cache
// of Apple M-series performance cores while every row of a streams past; wider
// tiles measured no faster.
const neonColumnTile = 8 * neonBlockColumns

// matMulTransposed is MatMulTransposed after its length checks, using NEON
// vector instructions on arm64. a ([m × k]), b ([n × k]), and out ([m × n])
// are row-major with lengths already checked by MatMulTransposed; out is
// fully overwritten and must not overlap a or b.
//
// Every output, whether it falls in a 4×5 block, a leftover row, or a leftover
// column, is computed by one procedure: a Float32x4 accumulator gathers fused
// multiply-adds over k in chunks of four, the final partial chunk is loaded
// zero-padded so it adds exact zeros, and the four lanes are reduced as
// (l0+l1)+(l2+l3). An output therefore depends only on its row of a, its row
// of b, and k, never on m or on how the caller groups rows.
func matMulTransposed(out, a, b []float32, m, k, n int) {
	for columnStart := 0; columnStart < n; columnStart += neonColumnTile {
		columnEnd := min(columnStart+neonColumnTile, n)
		row := 0
		for ; row+neonBlockRows <= m; row += neonBlockRows {
			neonFourRows(out, a, b, row, columnStart, columnEnd, k, n)
		}
		for ; row < m; row++ {
			neonOneRow(out, a, b, row, columnStart, columnEnd, k, n)
		}
	}
}

// neonFourRows fills out rows [row, row+4) across columns [columnStart,
// columnEnd), using the 4×5 micro-kernel where five columns remain and
// neonFourByOne for the rest.
func neonFourRows(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	a0 := a[row*k : (row+1)*k]
	a1 := a[(row+1)*k : (row+2)*k]
	a2 := a[(row+2)*k : (row+3)*k]
	a3 := a[(row+3)*k : (row+4)*k]
	column := columnStart
	for ; column+neonBlockColumns <= columnEnd; column += neonBlockColumns {
		r0, r1, r2, r3, lastColumn := neonFourByFive(a0, a1, a2, a3,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k],
			b[(column+4)*k:(column+5)*k])
		r0.StoreArray((*[neonLanes]float32)(out[row*n+column:]))
		r1.StoreArray((*[neonLanes]float32)(out[(row+1)*n+column:]))
		r2.StoreArray((*[neonLanes]float32)(out[(row+2)*n+column:]))
		r3.StoreArray((*[neonLanes]float32)(out[(row+3)*n+column:]))
		storeColumn(out, lastColumn, row*n+column+neonLanes, n)
	}
	for ; column < columnEnd; column++ {
		storeColumn(out, neonFourByOne(a0, a1, a2, a3, b[column*k:(column+1)*k]), row*n+column, n)
	}
}

// neonOneRow fills out row row across columns [columnStart, columnEnd); it
// handles the rows left over when m is not a multiple of four, four columns
// at a time.
func neonOneRow(out, a, b []float32, row, columnStart, columnEnd, k, n int) {
	aRow := a[row*k : (row+1)*k]
	outRow := out[row*n : (row+1)*n]
	column := columnStart
	for ; column+neonLanes <= columnEnd; column += neonLanes {
		neonOneByFour(aRow,
			b[column*k:(column+1)*k], b[(column+1)*k:(column+2)*k],
			b[(column+2)*k:(column+3)*k], b[(column+3)*k:(column+4)*k],
		).StoreArray((*[neonLanes]float32)(outRow[column:]))
	}
	for ; column < columnEnd; column++ {
		outRow[column] = neonOneByOne(aRow, b[column*k:(column+1)*k])
	}
}

// storeColumn writes the four lanes of sums down one column of out: lane i
// goes to out[start+i·n], where n is the row stride of out.
func storeColumn(out []float32, sums archsimd.Float32x4, start, n int) {
	var lanes [neonLanes]float32
	sums.StoreArray(&lanes)
	for i, value := range lanes {
		out[start+i*n] = value
	}
}

// reduceFour returns the lane sums of c0, c1, c2, and c3 in lanes 0 through 3,
// each reduced as (l0+l1)+(l2+l3). It is the only lane reduction the NEON
// kernel uses, which keeps every output's rounding identical across paths.
func reduceFour(c0, c1, c2, c3 archsimd.Float32x4) archsimd.Float32x4 {
	return c0.ConcatAddPairs(c1).ConcatAddPairs(c2.ConcatAddPairs(c3))
}

// loadFull returns s[p:p+4] as a vector; p+4 must not exceed len(s).
func loadFull(s []float32, p int) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[neonLanes]float32)(s[p : p+neonLanes]))
}

// loadPart returns the final partial chunk s[p:] as a vector with its missing
// lanes zeroed, so they contribute exact zeros to a fused multiply-add. p must
// be less than len(s).
func loadPart(s []float32, p int) archsimd.Float32x4 {
	chunk, _ := archsimd.LoadFloat32x4Part(s[p:])
	return chunk
}
