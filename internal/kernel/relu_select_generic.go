//go:build !goexperiment.simd || !(arm64 || amd64)

package kernel

// relu is ReLU. Builds without a SIMD implementation for the target
// architecture use the generic code.
func relu(x []float32) {
	reluGeneric(x)
}
