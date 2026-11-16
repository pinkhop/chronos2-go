package kernel

// Add writes the element-wise sum a + b into out, as used by residual
// connections.
//
// out may alias a or b, which accumulates in place. It panics unless out, a,
// and b share one length.
func Add(out, a, b []float32) {
	clearUpperBits()
	requireLength("a", len(a), len(out))
	requireLength("b", len(b), len(out))
	a = a[:len(out)]
	b = b[:len(out)]
	for i := range out {
		out[i] = a[i] + b[i]
	}
}
