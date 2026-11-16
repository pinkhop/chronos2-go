//go:build !goexperiment.simd || !(arm64 || amd64)

package kernel

// softmaxRows is SoftmaxRows after its length check. Builds without a SIMD
// implementation for the target architecture use the generic code.
func softmaxRows(x []float32, rows, cols int) {
	softmaxRowsGeneric(x, rows, cols)
}
