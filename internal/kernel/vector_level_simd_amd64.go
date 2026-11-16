//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// vectorLevel names the widest amd64 vector instruction set the SIMD kernels
// may use on the running CPU.
type vectorLevel int

const (
	// vectorLevelUnspecified is the zero value; no detected CPU reports it.
	vectorLevelUnspecified vectorLevel = iota
	// vectorLevelGeneric means the CPU lacks AVX2 or FMA, so the kernels
	// fall back to their generic implementations.
	vectorLevelGeneric
	// vectorLevelAVX2 means the CPU supports AVX2 and FMA: 8-lane kernels.
	vectorLevelAVX2
	// vectorLevelAVX512 means the CPU supports AVX-512 (F, CD, BW, DQ, and
	// VL) as well as AVX2 and FMA: 16-lane kernels.
	vectorLevelAVX512
)

// detectedVectorLevel is the vector level of the running CPU, detected once
// at package initialization and never modified, so every call in a process
// uses the same implementation and therefore the same rounding.
var detectedVectorLevel = detectVectorLevel()

// detectVectorLevel reports the widest vector level the running CPU and
// operating system support. archsimd's feature checks include the operating
// system's support for saving the wide registers.
func detectVectorLevel() vectorLevel {
	switch {
	case archsimd.X86.AVX512() && archsimd.X86.FMA():
		return vectorLevelAVX512
	case archsimd.X86.AVX2() && archsimd.X86.FMA():
		return vectorLevelAVX2
	}
	return vectorLevelGeneric
}
