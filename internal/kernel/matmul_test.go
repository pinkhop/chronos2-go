package kernel

import "testing"

// referenceMatMulTransposed computes a[m×k] · b[n×k]ᵀ in float64.
func referenceMatMulTransposed(a, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var sum float64
			for p := range k {
				sum += float64(a[i*k+p]) * float64(b[j*k+p])
			}
			out[i*n+j] = sum
		}
	}
	return out
}

// referenceMatMul computes a[m×k] · b[k×n] in float64.
func referenceMatMul(a, b []float32, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var sum float64
			for p := range k {
				sum += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			out[i*n+j] = sum
		}
	}
	return out
}

// matMulShapes covers degenerate, odd, and micro-kernel-remainder shapes.
var matMulShapes = []struct {
	name    string
	m, k, n int
}{
	{"one by one", 1, 1, 1},
	{"odd sizes", 3, 5, 7},
	{"remainders", 5, 13, 11},
	{"block multiples", 8, 64, 16},
	{"wider than a block", 9, 70, 133},
}

// TestMatMulTransposed_VariousShapes_MatchesReference verifies that MatMulTransposed, given various shapes, matches reference.
func TestMatMulTransposed_VariousShapes_MatchesReference(t *testing.T) {
	t.Parallel()
	for _, shape := range matMulShapes {
		t.Run(shape.name, func(t *testing.T) {
			t.Parallel()

			// Given
			a := randomValues(1, shape.m*shape.k)
			b := randomValues(2, shape.n*shape.k)
			out := make([]float32, shape.m*shape.n)

			// When
			MatMulTransposed(out, a, b, shape.m, shape.k, shape.n)

			// Then
			assertClose(t, "out", out, referenceMatMulTransposed(a, b, shape.m, shape.k, shape.n))
		})
	}
}

// TestMatMulTransposed_StaleOutput_IsOverwritten verifies that MatMulTransposed, given stale output, is overwritten.
func TestMatMulTransposed_StaleOutput_IsOverwritten(t *testing.T) {
	t.Parallel()

	// Given
	a := []float32{1, 2}
	b := []float32{3, 4}
	out := []float32{99}

	// When
	MatMulTransposed(out, a, b, 1, 2, 1)

	// Then
	if out[0] != 11 {
		t.Errorf("out[0] = %g, want 11", out[0])
	}
}

// TestMatMulTransposed_ShortOutput_Panics verifies that MatMulTransposed, given short output, panics.
func TestMatMulTransposed_ShortOutput_Panics(t *testing.T) {
	t.Parallel()

	// Given
	a := make([]float32, 6)
	b := make([]float32, 6)
	out := make([]float32, 3)

	// When / Then
	assertPanics(t, func() { MatMulTransposed(out, a, b, 2, 3, 2) })
}

// TestMatMul_VariousShapes_MatchesReference verifies that MatMul, given various shapes, matches reference.
func TestMatMul_VariousShapes_MatchesReference(t *testing.T) {
	t.Parallel()
	for _, shape := range matMulShapes {
		t.Run(shape.name, func(t *testing.T) {
			t.Parallel()

			// Given
			a := randomValues(3, shape.m*shape.k)
			b := randomValues(4, shape.k*shape.n)
			out := make([]float32, shape.m*shape.n)

			// When
			MatMul(out, a, b, shape.m, shape.k, shape.n)

			// Then
			assertClose(t, "out", out, referenceMatMul(a, b, shape.m, shape.k, shape.n))
		})
	}
}

// TestMatMul_StaleOutput_IsOverwritten verifies that MatMul, given stale output, is overwritten.
func TestMatMul_StaleOutput_IsOverwritten(t *testing.T) {
	t.Parallel()

	// Given
	a := []float32{1, 2}
	b := []float32{3, 4}
	out := []float32{99}

	// When
	MatMul(out, a, b, 1, 2, 1)

	// Then
	if out[0] != 11 {
		t.Errorf("out[0] = %g, want 11", out[0])
	}
}

// TestMatMul_ShortRightOperand_Panics verifies that MatMul, given short right operand, panics.
func TestMatMul_ShortRightOperand_Panics(t *testing.T) {
	t.Parallel()

	// Given
	a := make([]float32, 6)
	b := make([]float32, 5)
	out := make([]float32, 4)

	// When / Then
	assertPanics(t, func() { MatMul(out, a, b, 2, 3, 2) })
}

// TestMatMulTransposed_RowsSplitAcrossCalls_GiveBitwiseIdenticalResults verifies that MatMulTransposed gives bitwise-identical results when rows are split across calls.
func TestMatMulTransposed_RowsSplitAcrossCalls_GiveBitwiseIdenticalResults(t *testing.T) {
	t.Parallel()

	// Given
	const m, k, n = 7, 37, 9
	a := randomValues(20, m*k)
	b := randomValues(21, n*k)
	whole := make([]float32, m*n)
	MatMulTransposed(whole, a, b, m, k, n)
	const split = 5
	tail := make([]float32, (m-split)*n)

	// When
	MatMulTransposed(tail, a[split*k:], b, m-split, k, n)

	// Then
	for i, value := range tail {
		if value != whole[split*n+i] {
			t.Fatalf("tail[%d] = %g, whole call gave %g", i, value, whole[split*n+i])
		}
	}
}
