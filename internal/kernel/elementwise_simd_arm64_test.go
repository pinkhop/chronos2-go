//go:build goexperiment.simd

package kernel

// simdReLUImplementations returns the NEON ReLU, which every arm64 CPU
// supports.
func simdReLUImplementations() []reluImplementation {
	return []reluImplementation{{name: "neon", apply: relu}}
}

// simdAddBiasImplementations returns the NEON AddBias, which every arm64 CPU
// supports.
func simdAddBiasImplementations() []addBiasImplementation {
	return []addBiasImplementation{{name: "neon", apply: addBias}}
}

// simdAddBiasReLUImplementations returns the NEON AddBiasReLU, which every
// arm64 CPU supports.
func simdAddBiasReLUImplementations() []addBiasImplementation {
	return []addBiasImplementation{{name: "neon", apply: addBiasReLU}}
}
