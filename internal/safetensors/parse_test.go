package safetensors_test

import (
	"encoding/binary"
	"errors"
	"maps"
	"slices"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/safetensors"
)

// TestParse_ValidFile_ReturnsTensorsAndMetadata verifies that Parse, given valid file, returns tensors and metadata.
func TestParse_ValidFile_ReturnsTensorsAndMetadata(t *testing.T) {
	t.Parallel()

	// Given
	blob := encodeFile(t, []testTensor{
		{name: "weight", dtype: "F32", shape: []int{2, 3}, data: float32Bytes(1, 2, 3, 4, 5, 6)},
		{name: "bias", dtype: "BF16", shape: []int{3}, data: make([]byte, 6)},
	}, map[string]string{"format": "pt"})

	// When
	file, err := safetensors.Parse(blob)
	// Then
	if err != nil {
		t.Fatalf("Parse() error = %v, want nil", err)
	}
	if got := slices.Sorted(maps.Keys(file.Tensors)); !slices.Equal(got, []string{"bias", "weight"}) {
		t.Errorf("tensor names = %v, want [bias weight]", got)
	}
	weight := file.Tensors["weight"]
	if weight.Name != "weight" || weight.DType != safetensors.F32 || !slices.Equal(weight.Shape, []int{2, 3}) {
		t.Errorf("weight = {%q %v %v}, want {weight F32 [2 3]}", weight.Name, weight.DType, weight.Shape)
	}
	if !slices.Equal(weight.Data, float32Bytes(1, 2, 3, 4, 5, 6)) {
		t.Errorf("weight data = %v, want the encoded payload", weight.Data)
	}
	if bias := file.Tensors["bias"]; bias.DType != safetensors.BF16 || len(bias.Data) != 6 {
		t.Errorf("bias = {%v, %d bytes}, want {BF16, 6 bytes}", bias.DType, len(bias.Data))
	}
	if !maps.Equal(file.Metadata, map[string]string{"format": "pt"}) {
		t.Errorf("metadata = %v, want map[format:pt]", file.Metadata)
	}
}

// TestParse_ValidFile_DataAliasesInput verifies that Parse returns tensor data that aliases its input.
func TestParse_ValidFile_DataAliasesInput(t *testing.T) {
	t.Parallel()

	// Given
	blob := encodeFile(t, []testTensor{
		{name: "a", dtype: "F32", shape: []int{1}, data: float32Bytes(1)},
		{name: "b", dtype: "F32", shape: []int{1}, data: float32Bytes(2)},
	}, nil)

	// When
	file, err := safetensors.Parse(blob)
	// Then
	if err != nil {
		t.Fatalf("Parse() error = %v, want nil", err)
	}
	a := file.Tensors["a"].Data
	if &a[0] != &blob[len(blob)-8] {
		t.Errorf("tensor a data does not alias the input")
	}
	if cap(a) != len(a) {
		t.Errorf("cap(a.Data) = %d, want %d so appends cannot overwrite tensor b", cap(a), len(a))
	}
}

// TestParse_NoMetadata_ReturnsEmptyMetadata verifies that Parse, given no metadata, returns empty metadata.
func TestParse_NoMetadata_ReturnsEmptyMetadata(t *testing.T) {
	t.Parallel()

	// Given
	blob := encodeFile(t, []testTensor{{name: "x", dtype: "I64", shape: []int{}, data: make([]byte, 8)}}, nil)

	// When
	file, err := safetensors.Parse(blob)
	// Then
	if err != nil {
		t.Fatalf("Parse() error = %v, want nil", err)
	}
	if len(file.Metadata) != 0 {
		t.Errorf("metadata = %v, want empty", file.Metadata)
	}
	if x := file.Tensors["x"]; len(x.Shape) != 0 || x.DType != safetensors.I64 {
		t.Errorf("scalar x = {%v %v}, want {I64 []}", x.DType, x.Shape)
	}
}

// TestParse_PaddedHeader_Accepted verifies that Parse, given padded header, is accepted.
func TestParse_PaddedHeader_Accepted(t *testing.T) {
	t.Parallel()

	// Given a header padded with trailing spaces, as the reference writer emits.
	header := []byte(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}    `)
	blob := assemble(header, float32Bytes(7))

	// When
	file, err := safetensors.Parse(blob)
	// Then
	if err != nil {
		t.Fatalf("Parse() error = %v, want nil", err)
	}
	if len(file.Tensors["x"].Data) != 4 {
		t.Errorf("len(x.Data) = %d, want 4", len(file.Tensors["x"].Data))
	}
}

// TestParse_EmptyTensor_Accepted verifies that Parse, given empty tensor, is accepted.
func TestParse_EmptyTensor_Accepted(t *testing.T) {
	t.Parallel()

	// Given
	blob := encodeFile(t, []testTensor{{name: "empty", dtype: "F32", shape: []int{0, 4}}}, nil)

	// When
	file, err := safetensors.Parse(blob)
	// Then
	if err != nil {
		t.Fatalf("Parse() error = %v, want nil", err)
	}
	if got := file.Tensors["empty"]; len(got.Data) != 0 || !slices.Equal(got.Shape, []int{0, 4}) {
		t.Errorf("empty = {%v, %d bytes}, want {[0 4], 0 bytes}", got.Shape, len(got.Data))
	}
}

// TestParse_MalformedInput_ReturnsSentinel verifies that Parse, given malformed input, returns sentinel.
func TestParse_MalformedInput_ReturnsSentinel(t *testing.T) {
	t.Parallel()

	tensorHeader := func(entry string) []byte { return []byte(`{"x":` + entry + `}`) }
	hugeLength := binary.LittleEndian.AppendUint64(nil, 1<<40)
	tests := []struct {
		name string
		blob []byte
		want error
	}{
		{"shorter than length prefix", []byte{1, 2, 3}, safetensors.ErrTruncated},
		{"header past end", append(binary.LittleEndian.AppendUint64(nil, 100), '{', '}'), safetensors.ErrTruncated},
		{"header length too large", append(hugeLength, '{', '}'), safetensors.ErrInvalidHeader},
		{"empty header", assemble(nil, nil), safetensors.ErrInvalidHeader},
		{"not JSON", assemble([]byte("{nope"), nil), safetensors.ErrInvalidHeader},
		{"not an object", assemble([]byte("[]"), nil), safetensors.ErrInvalidHeader},
		{"trailing data after object", assemble([]byte("{} {}"), nil), safetensors.ErrInvalidHeader},
		{"duplicate tensor", assemble([]byte(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"entry not an object", assemble(tensorHeader(`5`), nil), safetensors.ErrInvalidHeader},
		{"missing dtype", assemble(tensorHeader(`{"shape":[1],"data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"missing shape", assemble(tensorHeader(`{"dtype":"F32","data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"missing offsets", assemble(tensorHeader(`{"dtype":"F32","shape":[1]}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"negative dimension", assemble(tensorHeader(`{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"one offset", assemble(tensorHeader(`{"dtype":"F32","shape":[1],"data_offsets":[4]}`), float32Bytes(1)), safetensors.ErrInvalidHeader},
		{"unknown dtype", assemble(tensorHeader(`{"dtype":"F31","shape":[1],"data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrUnsupportedDType},
		{"metadata not strings", assemble([]byte(`{"__metadata__":{"a":1}}`), nil), safetensors.ErrInvalidHeader},
		{"size mismatch", assemble(tensorHeader(`{"dtype":"F32","shape":[2],"data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrInvalidOffsets},
		{"shape overflow", assemble(tensorHeader(`{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}`), float32Bytes(1)), safetensors.ErrInvalidOffsets},
		{"reversed offsets", assemble(tensorHeader(`{"dtype":"F32","shape":[0],"data_offsets":[4,0]}`), float32Bytes(1)), safetensors.ErrInvalidOffsets},
		{"end out of bounds", assemble(tensorHeader(`{"dtype":"F32","shape":[2],"data_offsets":[0,8]}`), float32Bytes(1)), safetensors.ErrInvalidOffsets},
		{"overlapping", assemble([]byte(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`), float32Bytes(1, 2)), safetensors.ErrInvalidOffsets},
		{"hole between tensors", assemble([]byte(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`), float32Bytes(1, 2, 3)), safetensors.ErrInvalidOffsets},
		{"trailing bytes", assemble(tensorHeader(`{"dtype":"F32","shape":[1],"data_offsets":[0,4]}`), float32Bytes(1, 2)), safetensors.ErrInvalidOffsets},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given tt.blob

			// When
			_, err := safetensors.Parse(tt.blob)

			// Then
			if !errors.Is(err, tt.want) {
				t.Errorf("Parse() error = %v, want errors.Is(err, %v)", err, tt.want)
			}
		})
	}
}
