package kernel

import "fmt"

// requireLength panics when a slice named name has length got instead of
// want. Kernels call it on entry so a shape error surfaces at the faulty
// call rather than as silent corruption or an out-of-range index deep inside
// a loop. It is shared by every implementation of the kernels.
func requireLength(name string, got, want int) {
	if got != want {
		panic(fmt.Sprintf("kernel: %s has length %d, want %d", name, got, want))
	}
}
