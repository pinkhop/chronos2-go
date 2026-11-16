package safetensors_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

// testTensor describes one tensor for encodeFile: its name, dtype name as it
// appears in the header, shape, and raw little-endian payload.
type testTensor struct {
	// name is the tensor's header key.
	name string

	// dtype is the header's dtype string, such as "F32"; it is written
	// verbatim, so tests can supply invalid names.
	dtype string

	// shape is the header's shape array.
	shape []int

	// data is the tensor's raw payload, appended to the data section in
	// declaration order.
	data []byte
}

// encodeFile builds a safetensors blob holding tensors in order, packed
// contiguously, with an optional "__metadata__" entry when metadata is
// non-nil. It fails the test if the header cannot be marshaled.
func encodeFile(t testing.TB, tensors []testTensor, metadata map[string]string) []byte {
	t.Helper()

	// Build the header with each tensor's offsets into the data section.
	header := make(map[string]any, len(tensors)+1)
	var payload []byte
	for _, tensor := range tensors {
		begin := len(payload)
		payload = append(payload, tensor.data...)
		header[tensor.name] = map[string]any{
			"dtype":        tensor.dtype,
			"shape":        tensor.shape,
			"data_offsets": []int{begin, len(payload)},
		}
	}
	if metadata != nil {
		header["__metadata__"] = metadata
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	return assemble(headerJSON, payload)
}

// assemble concatenates the 8-byte little-endian header length, headerJSON,
// and payload into one safetensors blob, so tests can hand-craft headers.
func assemble(headerJSON, payload []byte) []byte {
	blob := binary.LittleEndian.AppendUint64(nil, uint64(len(headerJSON)))
	blob = append(blob, headerJSON...)
	return append(blob, payload...)
}

// float32Bytes encodes values as little-endian IEEE 754 binary32.
func float32Bytes(values ...float32) []byte {
	encoded := make([]byte, 0, 4*len(values))
	for _, value := range values {
		encoded = binary.LittleEndian.AppendUint32(encoded, math.Float32bits(value))
	}
	return encoded
}
