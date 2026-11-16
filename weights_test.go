package chronos2

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

// shardTensor describes one tensor for encodeShard.
type shardTensor struct {
	// name is the tensor's header key.
	name string

	// dtype is the header's dtype name.
	dtype string

	// shape is the tensor's shape.
	shape []int

	// data is the raw payload.
	data []byte
}

// encodeShard builds a safetensors shard whose header is padded with spaces
// to a multiple of 8 bytes, as the reference writer does, so the data section
// starts 8-byte aligned relative to the shard's start.
func encodeShard(t *testing.T, tensors ...shardTensor) []byte {
	t.Helper()
	header := map[string]any{}
	var payload []byte
	for _, tensor := range tensors {
		begin := len(payload)
		payload = append(payload, tensor.data...)
		header[tensor.name] = map[string]any{"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": []int{begin, len(payload)}}
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	for len(headerJSON)%8 != 0 {
		headerJSON = append(headerJSON, ' ')
	}
	blob := binary.LittleEndian.AppendUint64(nil, uint64(len(headerJSON)))
	blob = append(blob, headerJSON...)
	return append(blob, payload...)
}

// f32 encodes values as little-endian float32 bytes.
func f32(values ...float32) []byte {
	encoded := make([]byte, 0, 4*len(values))
	for _, value := range values {
		encoded = binary.LittleEndian.AppendUint32(encoded, math.Float32bits(value))
	}
	return encoded
}

// TestBindTensors_AlignedShard_AliasesShardBytes verifies that BindTensors, given aligned shard, aliases shard bytes.
func TestBindTensors_AlignedShard_AliasesShardBytes(t *testing.T) {
	t.Parallel()

	// Given a shard whose data section is 4-byte aligned in memory.
	shard := encodeShard(t, shardTensor{name: "w", dtype: "F32", shape: []int{2, 2}, data: f32(1, 2, 3, 4)})
	if !nativeLittleEndian {
		t.Skip("aliasing requires a little-endian platform")
	}

	// When
	tensors, err := bindTensors([][]byte{shard})
	// Then
	if err != nil {
		t.Fatalf("bindTensors() error = %v", err)
	}
	w := tensors["w"]
	if !slices.Equal(w.data, []float32{1, 2, 3, 4}) || !slices.Equal(w.shape, []int{2, 2}) {
		t.Errorf("w = %v %v, want [1 2 3 4] [2 2]", w.data, w.shape)
	}
	if got, want := fmt.Sprintf("%p", &w.data[0]), fmt.Sprintf("%p", &shard[len(shard)-16]); got != want {
		t.Errorf("w.data at %s, want the shard's bytes at %s", got, want)
	}
}

// TestBindTensors_MisalignedShard_CopiesValues verifies that BindTensors, given misaligned shard, copies values.
func TestBindTensors_MisalignedShard_CopiesValues(t *testing.T) {
	t.Parallel()

	// Given a shard that starts one byte past an aligned address, so its data
	// section is misaligned.
	shard := append([]byte{'x'}, encodeShard(t, shardTensor{name: "w", dtype: "F32", shape: []int{3}, data: f32(5, 6, 7)})...)[1:]

	// When
	tensors, err := bindTensors([][]byte{shard})
	// Then
	if err != nil {
		t.Fatalf("bindTensors() error = %v", err)
	}
	w := tensors["w"]
	if !slices.Equal(w.data, []float32{5, 6, 7}) {
		t.Errorf("w = %v, want [5 6 7]", w.data)
	}
	if got, aliased := fmt.Sprintf("%p", &w.data[0]), fmt.Sprintf("%p", &shard[len(shard)-12]); got == aliased {
		t.Errorf("w.data aliases misaligned shard bytes at %s", got)
	}
}

// TestBindTensors_InvalidShards_ReturnsInvalidCheckpoint verifies that BindTensors, given invalid shards, returns invalid checkpoint.
func TestBindTensors_InvalidShards_ReturnsInvalidCheckpoint(t *testing.T) {
	t.Parallel()

	weight := shardTensor{name: "w", dtype: "F32", shape: []int{1}, data: f32(1)}
	tests := []struct {
		name   string
		shards func(t *testing.T) [][]byte
	}{
		{"no shards", func(*testing.T) [][]byte { return nil }},
		{"malformed shard", func(*testing.T) [][]byte { return [][]byte{[]byte("not safetensors")} }},
		{"non-F32 tensor", func(t *testing.T) [][]byte {
			return [][]byte{encodeShard(t, shardTensor{name: "h", dtype: "F16", shape: []int{2}, data: make([]byte, 4)})}
		}},
		{"duplicate across shards", func(t *testing.T) [][]byte {
			return [][]byte{encodeShard(t, weight), encodeShard(t, weight)}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Given
			shards := tt.shards(t)

			// When
			_, err := bindTensors(shards)

			// Then
			if !errors.Is(err, ErrInvalidCheckpoint) {
				t.Errorf("bindTensors() error = %v, want ErrInvalidCheckpoint", err)
			}
		})
	}
}

// TestLookup_KnownName_ReturnsTensor verifies that Lookup, given known name, returns tensor.
func TestLookup_KnownName_ReturnsTensor(t *testing.T) {
	t.Parallel()

	// Given
	find := lookup(map[string]boundTensor{"w": {data: []float32{1}, shape: []int{1}}})

	// When
	data, shape, found := find("w")

	// Then
	if !found || !slices.Equal(data, []float32{1}) || !slices.Equal(shape, []int{1}) {
		t.Errorf("lookup(w) = %v %v %v, want [1] [1] true", data, shape, found)
	}
}

// TestLookup_UnknownName_ReportsMissing verifies that Lookup, given unknown name, reports missing.
func TestLookup_UnknownName_ReportsMissing(t *testing.T) {
	t.Parallel()

	// Given
	find := lookup(map[string]boundTensor{"w": {data: []float32{1}, shape: []int{1}}})

	// When
	_, _, found := find("absent")

	// Then
	if found {
		t.Errorf("lookup(absent) found = true, want false")
	}
}
