//go:build goexperiment.simd && (amd64 || arm64)

package kernel

import (
	"fmt"
	"math"
	"testing"
)

// softmaxRowsImplementation is one SIMD implementation of SoftmaxRows
// (after its length check) that the running CPU supports.
type softmaxRowsImplementation struct {
	// name labels the implementation in subtest names.
	name string
	// normalize replaces each row of x with its softmax in place.
	normalize func(x []float32, rows, cols int)
}

// vectorExpImplementation is one SIMD softmax exponential that the running
// CPU supports.
type vectorExpImplementation struct {
	// name labels the implementation in subtest names.
	name string
	// exponentiate returns eˣ for each input, evaluating the vector
	// exponential a chunk at a time with the final chunk zero-padded.
	exponentiate func(inputs []float32) []float32
}

// maxExpULPError is the largest acceptable error of a vector exponential, in
// units in the last place of the correctly rounded float32 result.
const maxExpULPError = 2

// expSweepStep is the spacing of the dense exponential accuracy sweep; 1/4096
// visits about 450,000 points across [-110, 0].
const expSweepStep = 1.0 / 4096

// smallestSubnormal is 2⁻¹⁴⁹, the spacing of float32 values below the
// smallest normal number.
var smallestSubnormal = math.Ldexp(1, -149)

// float32ULP returns the spacing of float32 values at the magnitude of value,
// the unit the exp accuracy bound is stated in; subnormal magnitudes share
// the fixed spacing 2⁻¹⁴⁹.
func float32ULP(value float64) float64 {
	if value == 0 {
		return smallestSubnormal
	}
	_, exponent := math.Frexp(value)
	return max(math.Ldexp(1, exponent-24), smallestSubnormal)
}

// TestVectorExp_DenseSweep_IsWithinTwoULP verifies that every vector
// exponential the CPU supports is within maxExpULPError of float64 math.Exp
// across [-110, 0], the only range softmax produces after subtracting the
// row maximum, including the subnormal results below about -87.3 and the
// zeros below about -103.97.
func TestVectorExp_DenseSweep_IsWithinTwoULP(t *testing.T) {
	t.Parallel()
	for _, implementation := range vectorExpImplementations() {
		t.Run(implementation.name, func(t *testing.T) {
			t.Parallel()

			// Given
			const points = int(110/expSweepStep) + 1
			inputs := make([]float32, points)
			for i := range inputs {
				inputs[i] = float32(-float64(i) * expSweepStep)
			}

			// When
			got := implementation.exponentiate(inputs)

			// Then
			worst, worstInput := 0.0, float32(0)
			for i, x := range inputs {
				want := math.Exp(float64(x))
				errorULP := math.Abs(float64(got[i])-want) / float32ULP(want)
				if errorULP > worst {
					worst, worstInput = errorULP, x
				}
			}
			t.Logf("largest error %.3f ulp at x = %g", worst, worstInput)
			if worst > maxExpULPError {
				t.Errorf("exp(%g) is %.3f ulp from math.Exp, want at most %d", worstInput, worst, maxExpULPError)
			}
		})
	}
}

// TestVectorExp_SpecialInputs_GiveExactResults verifies that every vector
// exponential the CPU supports returns exactly 1 for zeros, exactly 0 for
// every input whose exponential rounds to zero (including masked scores and
// negative infinity), and NaN for NaN.
func TestVectorExp_SpecialInputs_GiveExactResults(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input float32
		want  float32
	}{
		{"zero", 0, 1},
		{"negative zero", float32(math.Copysign(0, -1)), 1},
		{"just past underflow", -103.98, 0},
		{"far below underflow", -1e30, 0},
		{"masked score", -math.MaxFloat32, 0},
		{"negative infinity", float32(math.Inf(-1)), 0},
		{"NaN", float32(math.NaN()), float32(math.NaN())},
	}
	for _, implementation := range vectorExpImplementations() {
		for _, test := range cases {
			t.Run(implementation.name+"/"+test.name, func(t *testing.T) {
				t.Parallel()

				// Given
				input := test.input

				// When
				got := implementation.exponentiate([]float32{input})[0]

				// Then
				if math.IsNaN(float64(test.want)) {
					if !math.IsNaN(float64(got)) {
						t.Errorf("exp(%g) = %g, want NaN", input, got)
					}
					return
				}
				if math.Float32bits(got) != math.Float32bits(test.want) {
					t.Errorf("exp(%g) = %g, want exactly %g", input, got, test.want)
				}
			})
		}
	}
}

// softmaxColumnCounts covers every tail length up to two full 8-lane chunks
// and four full 4-lane chunks, the attention-head width, and the full
// 8192-step attention sequence with and without a leftover element.
var softmaxColumnCounts = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 64, 577, 578}

// softmaxRowKinds is the number of rows mixedSoftmaxRows builds.
const softmaxRowKinds = 7

// mixedSoftmaxRows returns softmaxRowKinds rows of cols scores each: random
// scores spanning about ±30, the same with every third score additively
// masked, every score masked, every score negative infinity, random scores
// with one NaN, random scores with one positive infinity, and negative
// infinity with one NaN.
func mixedSoftmaxRows(cols int) []float32 {
	const masked = -math.MaxFloat32
	x := randomValues(uint64(cols), softmaxRowKinds*cols)
	for i := range x {
		x[i] *= 30
	}
	for c := range cols {
		if c%3 == 1 {
			x[cols+c] = masked
		}
		x[2*cols+c] = masked
		x[3*cols+c] = float32(math.Inf(-1))
		x[6*cols+c] = float32(math.Inf(-1))
	}
	x[4*cols+cols/2] = float32(math.NaN())
	x[5*cols+cols/2] = float32(math.Inf(1))
	x[6*cols+cols/2] = float32(math.NaN())
	return x
}

// assertSoftmaxClose fails the test when any element of got differs from
// want by more than the kernel tolerances, or when exactly one is NaN.
func assertSoftmaxClose(t *testing.T, got, want []float32) {
	t.Helper()
	for i := range want {
		gotNaN, wantNaN := math.IsNaN(float64(got[i])), math.IsNaN(float64(want[i]))
		if gotNaN != wantNaN {
			t.Errorf("x[%d] = %g, want %g", i, got[i], want[i])
			return
		}
		difference := math.Abs(float64(got[i]) - float64(want[i]))
		if !wantNaN && difference > absoluteTolerance+relativeTolerance*math.Abs(float64(want[i])) {
			t.Errorf("x[%d] = %g, want %g", i, got[i], want[i])
			return
		}
	}
}

// TestSoftmaxRows_SIMDMixedRows_MatchesGeneric verifies that every SIMD
// softmaxRows the CPU supports agrees with softmaxRowsGeneric within float32
// rounding for random, partly masked, fully masked, negative-infinity, NaN,
// and positive-infinity rows at every tail length.
func TestSoftmaxRows_SIMDMixedRows_MatchesGeneric(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdSoftmaxRowsImplementations() {
		for _, cols := range softmaxColumnCounts {
			t.Run(fmt.Sprintf("%s/cols=%d", implementation.name, cols), func(t *testing.T) {
				t.Parallel()

				// Given
				want := mixedSoftmaxRows(cols)
				softmaxRowsGeneric(want, softmaxRowKinds, cols)
				got := mixedSoftmaxRows(cols)

				// When
				implementation.normalize(got, softmaxRowKinds, cols)

				// Then
				assertSoftmaxClose(t, got, want)
			})
		}
	}
}

// TestSoftmaxRows_SIMDEveryKeyMasked_MatchesGenericBitwise verifies that a
// fully masked row gets exactly the generic build's weights, 1·(1/cols),
// which the network's singleton-group shortcut relies on.
func TestSoftmaxRows_SIMDEveryKeyMasked_MatchesGenericBitwise(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdSoftmaxRowsImplementations() {
		for _, cols := range softmaxColumnCounts {
			t.Run(fmt.Sprintf("%s/cols=%d", implementation.name, cols), func(t *testing.T) {
				t.Parallel()

				// Given
				want := make([]float32, cols)
				for i := range want {
					want[i] = -math.MaxFloat32
				}
				got := append([]float32(nil), want...)
				softmaxRowsGeneric(want, 1, cols)

				// When
				implementation.normalize(got, 1, cols)

				// Then
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("x[%d] = %g, generic gave %g", i, got[i], want[i])
					}
				}
			})
		}
	}
}
