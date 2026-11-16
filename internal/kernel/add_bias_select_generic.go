//go:build !goexperiment.simd || !(arm64 || amd64)

package kernel

// addBias is AddBias after its length checks. Builds without a SIMD
// implementation for the target architecture use the generic code.
func addBias(x, bias []float32, rows, cols int) {
	addBiasGeneric(x, bias, rows, cols)
}
