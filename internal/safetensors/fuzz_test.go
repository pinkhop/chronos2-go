package safetensors_test

import (
	"errors"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// FuzzParse checks that Parse never panics, that every failure matches a
// package sentinel, and that every accepted tensor's data is exactly the
// size its dtype and shape imply.
func FuzzParse(f *testing.F) {
	f.Add(encodeFile(f, []testTensor{
		{name: "w", dtype: "F32", shape: []int{2}, data: float32Bytes(1, 2)},
		{name: "b", dtype: "U8", shape: []int{3}, data: []byte{1, 2, 3}},
	}, map[string]string{"format": "pt"}))
	f.Add(assemble([]byte(`{}`), nil))
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, blob []byte) {
		// Given blob

		// When
		file, err := safetensors.Parse(blob)
		// Then
		if err != nil {
			if !isSentinel(err) {
				t.Errorf("Parse() error = %v, matches no sentinel", err)
			}
			return
		}
		for name, tensor := range file.Tensors {
			elements := 1
			for _, dimension := range tensor.Shape {
				elements *= dimension
			}
			if len(tensor.Data) != elements*tensor.DType.Size() {
				t.Errorf("tensor %q: %d bytes, want %d", name, len(tensor.Data), elements*tensor.DType.Size())
			}
		}
	})
}

// isSentinel reports whether err matches one of the errors Parse documents.
func isSentinel(err error) bool {
	return errors.Is(err, safetensors.ErrTruncated) ||
		errors.Is(err, safetensors.ErrInvalidHeader) ||
		errors.Is(err, safetensors.ErrUnsupportedDType) ||
		errors.Is(err, safetensors.ErrInvalidOffsets)
}
