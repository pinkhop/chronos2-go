//go:build goexperiment.simd

package kernel

// addBias is AddBias after its length checks, using NEON instructions four
// elements at a time, with the elements after each row's last full vector
// handled by addBiasRowGeneric. Each element gets the same single float32
// addition as in the generic code.
func addBias(x, bias []float32, rows, cols int) {
	for r := range rows {
		row := x[r*cols : (r+1)*cols]
		c := 0
		for ; c+neonLanes <= cols; c += neonLanes {
			sum := loadFull(row, c).Add(loadFull(bias, c))
			sum.StoreArray((*[neonLanes]float32)(row[c : c+neonLanes]))
		}
		addBiasRowGeneric(row[c:], bias[c:])
	}
}
