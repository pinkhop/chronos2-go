//go:build !goexperiment.simd || !(arm64 || amd64)

package kernel

// addBiasReLU is AddBiasReLU after its length checks. Builds without a SIMD
// implementation for the target architecture use the generic code.
func addBiasReLU(x, bias []float32, rows, cols int) {
	addBiasReLUGeneric(x, bias, rows, cols)
}
