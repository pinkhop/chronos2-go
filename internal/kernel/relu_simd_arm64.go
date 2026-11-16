//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// relu is ReLU using NEON instructions, four elements at a time, with the
// elements after the last full vector handled by reluGeneric. It keeps a lane
// where the value is positive or NaN (the only value unequal to itself) and
// writes +0 elsewhere, which is exactly the built-in max(value, 0) the
// generic code uses.
func relu(x []float32) {
	var zero archsimd.Float32x4
	p := 0
	for ; p+neonLanes <= len(x); p += neonLanes {
		value := loadFull(x, p)
		keep := value.Greater(zero).Or(value.NotEqual(value))
		value.Masked(keep).StoreArray((*[neonLanes]float32)(x[p : p+neonLanes]))
	}
	reluGeneric(x[p:])
}
