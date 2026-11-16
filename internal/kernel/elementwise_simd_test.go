//go:build goexperiment.simd && (amd64 || arm64)

package kernel

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// reluImplementation is one SIMD implementation of ReLU that the running CPU
// supports.
type reluImplementation struct {
	// name labels the implementation in subtest names.
	name string
	// apply clamps x in place.
	apply func(x []float32)
}

// addBiasImplementation is one SIMD implementation of AddBias or AddBiasReLU
// (after its length checks) that the running CPU supports.
type addBiasImplementation struct {
	// name labels the implementation in subtest names.
	name string
	// apply adds bias to every row of x in place, clamping the sums at zero
	// for AddBiasReLU.
	apply func(x, bias []float32, rows, cols int)
}

// elementwiseLengths covers empty input, every tail length beside 4- and
// 8-lane vectors, and several full vectors with and without a tail.
var elementwiseLengths = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 31, 32, 33, 64, 577}

// specialElementValues returns count values that cycle through random
// values, both zeros, both infinities, NaN, and the smallest subnormals of
// each sign, so every lane position meets every special value.
func specialElementValues(seed uint64, count int) []float32 {
	values := randomValues(seed, count)
	specials := []float32{
		0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
	}
	for i := 0; i < count; i += 3 {
		values[i] = specials[(i/3)%len(specials)]
	}
	return values
}

// assertSameBits fails the test unless got and want hold identical bits,
// so negative zero and every rounding must match exactly. Any NaN matches any
// other NaN: x86 arithmetic produces a negative default NaN that Go's
// built-in max replaces with a positive one, and no caller distinguishes NaN
// payloads.
func assertSameBits(t *testing.T, got, want []float32) {
	t.Helper()
	for i := range want {
		bothNaN := got[i] != got[i] && want[i] != want[i]
		if !bothNaN && math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("x[%d] = %g (%#08x), want %g (%#08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// TestReLU_SIMDSpecialValues_MatchesGenericBitwise verifies that every SIMD
// ReLU the CPU supports gives exactly the generic result, including NaN,
// negative zero, infinities, and subnormals, at every tail length.
func TestReLU_SIMDSpecialValues_MatchesGenericBitwise(t *testing.T) {
	t.Parallel()
	for _, implementation := range simdReLUImplementations() {
		for _, length := range elementwiseLengths {
			t.Run(fmt.Sprintf("%s/length=%d", implementation.name, length), func(t *testing.T) {
				t.Parallel()

				// Given
				want := specialElementValues(uint64(length)+60, length)
				got := slices.Clone(want)
				reluGeneric(want)

				// When
				implementation.apply(got)

				// Then
				assertSameBits(t, got, want)
			})
		}
	}
}

// TestAddBias_SIMDSpecialValues_MatchesGenericBitwise verifies that every
// SIMD AddBias the CPU supports gives exactly the generic result, including
// NaN, signed zeros, infinities, and subnormals, at every row tail length.
func TestAddBias_SIMDSpecialValues_MatchesGenericBitwise(t *testing.T) {
	t.Parallel()
	const rows = 3
	for _, implementation := range simdAddBiasImplementations() {
		for _, cols := range elementwiseLengths {
			t.Run(fmt.Sprintf("%s/cols=%d", implementation.name, cols), func(t *testing.T) {
				t.Parallel()

				// Given
				want := specialElementValues(uint64(cols)+70, rows*cols)
				got := slices.Clone(want)
				bias := specialElementValues(uint64(cols)+80, cols)
				slices.Reverse(bias)
				addBiasGeneric(want, bias, rows, cols)

				// When
				implementation.apply(got, bias, rows, cols)

				// Then
				assertSameBits(t, got, want)
			})
		}
	}
}

// TestAddBiasReLU_SIMDSpecialValues_MatchesGenericBitwise verifies that every
// SIMD AddBiasReLU the CPU supports gives exactly the generic result,
// including NaN, signed zeros, infinities, and subnormals, at every row tail
// length.
func TestAddBiasReLU_SIMDSpecialValues_MatchesGenericBitwise(t *testing.T) {
	t.Parallel()
	const rows = 3
	for _, implementation := range simdAddBiasReLUImplementations() {
		for _, cols := range elementwiseLengths {
			t.Run(fmt.Sprintf("%s/cols=%d", implementation.name, cols), func(t *testing.T) {
				t.Parallel()

				// Given
				want := specialElementValues(uint64(cols)+90, rows*cols)
				got := slices.Clone(want)
				bias := specialElementValues(uint64(cols)+100, cols)
				slices.Reverse(bias)
				addBiasReLUGeneric(want, bias, rows, cols)

				// When
				implementation.apply(got, bias, rows, cols)

				// Then
				assertSameBits(t, got, want)
			})
		}
	}
}
