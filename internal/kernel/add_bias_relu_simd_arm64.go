//go:build goexperiment.simd

package kernel

import "simd/archsimd"

// addBiasReLU is AddBiasReLU after its length checks, using NEON
// instructions four elements at a time, with the elements after each row's
// last full vector handled by addBiasReLURowGeneric. Each sum is the same
// single float32 addition as in the generic code, and the clamp keeps exactly
// the lanes relu keeps.
func addBiasReLU(x, bias []float32, rows, cols int) {
	var zero archsimd.Float32x4
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		c := 0
		for ; c+neonLanes <= cols; c += neonLanes {
			sum := loadFull(row, c).Add(loadFull(bias, c))
			keep := sum.Greater(zero).Or(sum.NotEqual(sum))
			sum.Masked(keep).StoreArray((*[neonLanes]float32)(row[c : c+neonLanes]))
		}
		addBiasReLURowGeneric(row[c:], bias[c:])
	}
}
