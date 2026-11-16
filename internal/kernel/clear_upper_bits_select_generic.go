//go:build !goexperiment.simd || !amd64

package kernel

// clearUpperBits does nothing outside the amd64 SIMD build, where scalar code
// never pays a penalty for dirty vector registers. See the amd64 SIMD
// implementation for why that build needs it.
func clearUpperBits() {}
