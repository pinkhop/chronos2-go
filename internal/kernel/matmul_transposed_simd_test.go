//go:build goexperiment.simd && (amd64 || arm64)

package kernel

import (
	"fmt"
	"math"
	"testing"
)

// matMulTransposedImplementation is one SIMD implementation of
// MatMulTransposed (after its length checks) that the running CPU supports.
type matMulTransposedImplementation struct {
	// name labels the implementation in subtest names.
	name string
	// multiply computes out = a · bᵀ with lengths already checked.
	multiply func(out, a, b []float32, m, k, n int)
}

// simdShape is one matrix shape exercised by the SIMD tests: a is m × k, b is
// n × k, and out is m × n.
type simdShape struct {
	// m is the number of rows of a and of out.
	m int
	// k is the shared inner dimension.
	k int
	// n is the number of rows of b and columns of out.
	n int
}

// String names the shape for subtest labels.
func (shape simdShape) String() string {
	return fmt.Sprintf("m=%d,k=%d,n=%d", shape.m, shape.k, shape.n)
}

// edgeRowCounts are values of m covering empty input, every count of leftover
// rows beside 2-, 3-, and 4-row blocks, and exact multiples of each block
// height.
var edgeRowCounts = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 13}

// edgeColumnCounts are values of n covering empty output, every remainder
// modulo the 4- and 5-column blocks, and the edges of the 32-, 40-, and
// 64-column tiles: exactly one and two tiles, one past a tile, and
// multi-tile counts ending in a partial tile.
var edgeColumnCounts = []int{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 31, 32, 33, 40, 41, 63, 64, 65, 71, 80, 97}

// edgeDepths are values of k covering an empty inner dimension, every
// partial final chunk of the 4-, 8-, and 16-lane vectors, and two or three
// full 16-lane chunks with and without a partial one.
var edgeDepths = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 47, 48}

// edgeShapes returns every combination of the edge depths, row counts, and
// column counts.
func edgeShapes() []simdShape {
	var shapes []simdShape
	for _, k := range edgeDepths {
		for _, m := range edgeRowCounts {
			for _, n := range edgeColumnCounts {
				shapes = append(shapes, simdShape{m: m, k: k, n: n})
			}
		}
	}
	return shapes
}

// staleOutput returns count NaN values, so any output a kernel fails to
// overwrite makes a comparison fail.
func staleOutput(count int) []float32 {
	out := make([]float32, count)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	return out
}

// widenToFloat64 converts values to float64 so a float32 result can serve as
// the expected value for assertClose.
func widenToFloat64(values []float32) []float64 {
	widened := make([]float64, len(values))
	for i, value := range values {
		widened[i] = float64(value)
	}
	return widened
}

// TestMatMulTransposed_SIMDEdgeShapes_MatchesGeneric verifies that every
// SIMD matMulTransposed the CPU supports overwrites every output and agrees
// with matMulTransposedGeneric within float32 rounding for shapes that leave
// remainders in every block and tile dimension, including empty matrices.
func TestMatMulTransposed_SIMDEdgeShapes_MatchesGeneric(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdMatMulTransposedImplementations() {
		for _, shape := range edgeShapes() {
			t.Run(implementation.name+"/"+shape.String(), func(t *testing.T) {
				t.Parallel()

				// Given
				a := randomValues(uint64(100*shape.k+shape.m), shape.m*shape.k)
				b := randomValues(uint64(1000*shape.k+shape.n), shape.n*shape.k)
				want := make([]float32, shape.m*shape.n)
				matMulTransposedGeneric(want, a, b, shape.m, shape.k, shape.n)
				got := staleOutput(shape.m * shape.n)

				// When
				implementation.multiply(got, a, b, shape.m, shape.k, shape.n)

				// Then
				assertClose(t, "out", got, widenToFloat64(want))
			})
		}
	}
}

// TestMatMulTransposed_SIMDModelWidths_MatchesReference verifies that every
// SIMD matMulTransposed the CPU supports matches the float64 reference at
// the attention-head width and at a width with a partial final chunk, where
// many chunks are accumulated per output.
func TestMatMulTransposed_SIMDModelWidths_MatchesReference(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdMatMulTransposedImplementations() {
		for _, shape := range []simdShape{{m: 13, k: headWidth, n: 71}, {m: 13, k: 131, n: 71}} {
			t.Run(implementation.name+"/"+shape.String(), func(t *testing.T) {
				t.Parallel()

				// Given
				a := randomValues(uint64(shape.k), shape.m*shape.k)
				b := randomValues(uint64(shape.k)+1, shape.n*shape.k)
				out := staleOutput(shape.m * shape.n)

				// When
				implementation.multiply(out, a, b, shape.m, shape.k, shape.n)

				// Then
				assertClose(t, "out", out, referenceMatMulTransposed(a, b, shape.m, shape.k, shape.n))
			})
		}
	}
}

// rowSplitShapes returns shapes whose rows fall in register blocks and
// leftover rows and whose columns fall in register blocks, leftover columns,
// and several tiles, across k values with and without a partial final chunk.
func rowSplitShapes() []simdShape {
	var shapes []simdShape
	for _, k := range []int{1, 3, 4, 7, 17, 33, 131} {
		for _, m := range []int{5, 7, 8, 13} {
			for _, n := range []int{9, 41, 71} {
				shapes = append(shapes, simdShape{m: m, k: k, n: n})
			}
		}
	}
	return shapes
}

// TestMatMulTransposed_SIMDEachRowAlone_GivesBitwiseIdenticalResults verifies
// that every SIMD matMulTransposed the CPU supports gives every row the same
// bits whether it is computed alone or inside a full call, so callers may
// split rows across goroutines without changing any value.
func TestMatMulTransposed_SIMDEachRowAlone_GivesBitwiseIdenticalResults(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdMatMulTransposedImplementations() {
		for _, shape := range rowSplitShapes() {
			t.Run(implementation.name+"/"+shape.String(), func(t *testing.T) {
				t.Parallel()

				// Given
				m, k, n := shape.m, shape.k, shape.n
				a := randomValues(uint64(k*m+n)+40, m*k)
				b := randomValues(uint64(k*m+n)+41, n*k)
				whole := make([]float32, m*n)
				implementation.multiply(whole, a, b, m, k, n)
				rows := staleOutput(m * n)

				// When
				for row := range m {
					implementation.multiply(rows[row*n:(row+1)*n], a[row*k:(row+1)*k], b, 1, k, n)
				}

				// Then
				for i, value := range rows {
					if value != whole[i] {
						t.Fatalf("row %d column %d alone = %g, whole call gave %g", i/n, i%n, value, whole[i])
					}
				}
			})
		}
	}
}
