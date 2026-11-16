package testmodel

import (
	"os"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// Tensor is one float32 fixture tensor.
type Tensor struct {
	// Shape lists the tensor's dimensions, outermost first.
	Shape []int
	// Values holds the elements in row-major order.
	Values []float32
}

// ReadTensors reads every tensor of a float32 safetensors fixture file, such
// as testdata/synthetic/forward_single.safetensors, keyed by tensor name. It
// fails the test when the file is missing, malformed, or holds a tensor of
// another dtype.
func ReadTensors(tb testing.TB, path string) map[string]Tensor {
	tb.Helper()
	// #nosec G304 -- tests pass fixed repository fixture paths; no untrusted input.
	data, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read fixture %s: %v", path, err)
	}
	file, err := safetensors.Parse(data)
	if err != nil {
		tb.Fatalf("parse fixture %s: %v", path, err)
	}
	tensors := make(map[string]Tensor, len(file.Tensors))
	for name, tensor := range file.Tensors {
		values, err := tensor.Float32s()
		if err != nil {
			tb.Fatalf("decode tensor %s in %s: %v", name, path, err)
		}
		tensors[name] = Tensor{Shape: tensor.Shape, Values: values}
	}
	return tensors
}

// Within reports whether got lies within the combined tolerance of want,
// |got − want| ≤ abs + rel·|want|, as DESIGN.md defines for parity checks.
// NaN matches only NaN.
func Within(got, want, abs, rel float64) bool {
	if got != got || want != want {
		return got != got && want != want
	}
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	magnitude := want
	if magnitude < 0 {
		magnitude = -magnitude
	}
	return diff <= abs+rel*magnitude
}
