package safetensors

import (
	"encoding/binary"
	"fmt"
	"math"
)

// float32Bytes is the width of one F32 element.
const float32Bytes = 4

// Tensor is a view of one tensor in a parsed safetensors file.
type Tensor struct {
	// Name is the tensor's key in the file header.
	Name string

	// DType is the tensor's element type; Parse never sets DTypeUnspecified.
	DType DType

	// Shape lists the tensor's dimensions, outermost first; it is empty for a
	// scalar.
	Shape []int

	// Data holds the tensor's raw little-endian elements in row-major order.
	// After Parse it aliases the parsed input and is never copied; callers
	// must not modify it.
	Data []byte
}

// File is the parsed content of a safetensors file.
type File struct {
	// Tensors maps each tensor's name to its view.
	Tensors map[string]Tensor

	// Metadata holds the header's "__metadata__" entry; it is nil or empty
	// when the file has none.
	Metadata map[string]string
}

// Float32s returns a newly allocated copy of an F32 tensor's elements, for
// callers that need values they may modify or that must outlive the parsed
// input. It returns an error wrapping ErrDTypeMismatch for any other dtype or
// ErrInvalidOffsets when Data is not a whole number of elements.
func (t Tensor) Float32s() ([]float32, error) {
	if t.DType != F32 {
		return nil, fmt.Errorf("%w: tensor %q is %v, want F32", ErrDTypeMismatch, t.Name, t.DType)
	}
	if len(t.Data)%float32Bytes != 0 {
		return nil, fmt.Errorf("%w: tensor %q holds %d bytes, not a multiple of %d", ErrInvalidOffsets, t.Name, len(t.Data), float32Bytes)
	}
	values := make([]float32, len(t.Data)/float32Bytes)
	DecodeFloat32(values, t.Data)
	return values, nil
}

// DecodeFloat32 decodes little-endian IEEE 754 binary32 values from src into
// dst, which callers use to fill a buffer they own from a tensor's Data. It
// panics unless len(src) == 4*len(dst), a programming error since Parse
// already ties each tensor's byte length to its shape.
func DecodeFloat32(dst []float32, src []byte) {
	if len(src) != float32Bytes*len(dst) {
		panic(fmt.Sprintf("safetensors: DecodeFloat32: %d source bytes for %d values", len(src), len(dst)))
	}
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[float32Bytes*i:]))
	}
}
