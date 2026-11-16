package safetensors_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// TestDecodeFloat32_LittleEndianBytes_DecodesValues verifies that DecodeFloat32, given little endian bytes, decodes values.
func TestDecodeFloat32_LittleEndianBytes_DecodesValues(t *testing.T) {
	t.Parallel()

	// Given
	src := float32Bytes(1.5, -2, float32(math.Inf(1)), 0)
	dst := make([]float32, 4)

	// When
	safetensors.DecodeFloat32(dst, src)

	// Then
	if want := []float32{1.5, -2, float32(math.Inf(1)), 0}; !slices.Equal(dst, want) {
		t.Errorf("dst = %v, want %v", dst, want)
	}
}

// TestDecodeFloat32_LengthMismatch_Panics verifies that DecodeFloat32, given length mismatch, panics.
func TestDecodeFloat32_LengthMismatch_Panics(t *testing.T) {
	t.Parallel()

	// Given
	src := float32Bytes(1, 2)
	dst := make([]float32, 3)

	// When
	recovered := func() (value any) {
		defer func() { value = recover() }()
		safetensors.DecodeFloat32(dst, src)
		return nil
	}()

	// Then
	if recovered == nil {
		t.Errorf("DecodeFloat32() did not panic on a length mismatch")
	}
}

// TestTensor_Float32s_F32Tensor_ReturnsCopy verifies that Tensor.Float32s, given F32 tensor, returns copy.
func TestTensor_Float32s_F32Tensor_ReturnsCopy(t *testing.T) {
	t.Parallel()

	// Given
	data := float32Bytes(3, 4)
	tensor := safetensors.Tensor{Name: "w", DType: safetensors.F32, Shape: []int{2}, Data: data}

	// When
	values, err := tensor.Float32s()
	// Then
	if err != nil {
		t.Fatalf("Float32s() error = %v, want nil", err)
	}
	if !slices.Equal(values, []float32{3, 4}) {
		t.Errorf("Float32s() = %v, want [3 4]", values)
	}
}

// TestTensor_Float32s_NonF32Tensor_ReturnsDTypeMismatch verifies that Tensor.Float32s, given non F32 tensor, returns DType mismatch.
func TestTensor_Float32s_NonF32Tensor_ReturnsDTypeMismatch(t *testing.T) {
	t.Parallel()

	// Given
	tensor := safetensors.Tensor{Name: "w", DType: safetensors.F16, Shape: []int{2}, Data: make([]byte, 4)}

	// When
	_, err := tensor.Float32s()

	// Then
	if !errors.Is(err, safetensors.ErrDTypeMismatch) {
		t.Errorf("Float32s() error = %v, want ErrDTypeMismatch", err)
	}
}

// TestDType_String_KnownAndUnspecified_ReturnsNames verifies that DType.String, given known and unspecified, returns names.
func TestDType_String_KnownAndUnspecified_ReturnsNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		dtype safetensors.DType
		want  string
	}{
		{safetensors.F32, "F32"},
		{safetensors.BF16, "BF16"},
		{safetensors.F8E4M3, "F8_E4M3"},
		{safetensors.DTypeUnspecified, "DTypeUnspecified"},
		{safetensors.DType(999), "DType(999)"},
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			t.Parallel()

			// Given tt.dtype

			// When
			got := tt.dtype.String()

			// Then
			if got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestDType_Size_F32AndUnspecified_ReturnsByteWidth verifies that DType.Size, given F32 and unspecified, returns byte width.
func TestDType_Size_F32AndUnspecified_ReturnsByteWidth(t *testing.T) {
	t.Parallel()

	tests := []struct {
		dtype safetensors.DType
		want  int
	}{
		{safetensors.F32, 4},
		{safetensors.F64, 8},
		{safetensors.BF16, 2},
		{safetensors.Bool, 1},
		{safetensors.DTypeUnspecified, 0},
	}
	for _, tt := range tests {
		t.Run(tt.dtype.String(), func(t *testing.T) {
			t.Parallel()

			// Given tt.dtype

			// When
			got := tt.dtype.Size()

			// Then
			if got != tt.want {
				t.Errorf("Size() = %d, want %d", got, tt.want)
			}
		})
	}
}
