//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// clearUpperBits clears the upper bits of the AVX registers (VZEROUPPER) so
// the scalar code that follows runs at full speed. Go compiles scalar float32
// arithmetic to legacy SSE instructions, which run several times slower on
// Intel CPUs while those bits hold data. The vector kernels leave them clear,
// but under GOEXPERIMENT=simd the runtime's asynchronous preemption restores
// every vector register in full, leaving them dirty at an arbitrary point in
// any goroutine. Every scalar kernel therefore calls clearUpperBits on entry,
// which bounds the slowdown to the one call a preemption interrupts. It does
// nothing on CPUs without AVX2, where the vector kernels never run and
// VZEROUPPER may not exist.
func clearUpperBits() {
	if detectedVectorLevel >= vectorLevelAVX2 {
		archsimd.ClearAVXUpperBits()
	}
}
