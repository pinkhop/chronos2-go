package safetensors

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// metadataKey is the reserved header key holding free-form string metadata
// rather than a tensor.
const metadataKey = "__metadata__"

// headerEntry is one raw top-level header member, kept in file order.
type headerEntry struct {
	// key is the member's name: a tensor name or metadataKey.
	key string

	// value is the member's undecoded JSON value.
	value json.RawMessage
}

// tensorDescription is a tensor's header entry. Pointer fields distinguish a
// missing field from a zero value.
type tensorDescription struct {
	// DType is the dtype name, such as "F32".
	DType *string `json:"dtype"`

	// Shape lists the tensor's dimensions; an empty list is a scalar.
	Shape *[]int `json:"shape"`

	// DataOffsets holds the tensor's [begin, end) byte range relative to the
	// start of the data section.
	DataOffsets *[]uint64 `json:"data_offsets"`
}

// splitHeader decodes headerJSON as one JSON object and returns its members in
// file order. Trailing whitespace is allowed, because writers pad the header
// with spaces for alignment. It returns an error wrapping ErrInvalidHeader
// when the header is not exactly one object or repeats a key.
func splitHeader(headerJSON []byte) ([]headerEntry, error) {
	decoder := json.NewDecoder(bytes.NewReader(headerJSON))
	if err := expectDelimiter(decoder, '{'); err != nil {
		return nil, err
	}

	// Read members one at a time so a repeated key, which encoding/json would
	// silently resolve to its last value, is rejected.
	var entries []headerEntry
	seen := make(map[string]bool)
	for decoder.More() {
		var entry headerEntry
		keyToken, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("%w: reading key: %w", ErrInvalidHeader, err)
		}
		entry.key, _ = keyToken.(string) // decoder.Token guarantees object keys are strings.
		if seen[entry.key] {
			return nil, fmt.Errorf("%w: duplicate key %q", ErrInvalidHeader, entry.key)
		}
		seen[entry.key] = true
		if err := decoder.Decode(&entry.value); err != nil {
			return nil, fmt.Errorf("%w: value of %q: %w", ErrInvalidHeader, entry.key, err)
		}
		entries = append(entries, entry)
	}
	if err := expectDelimiter(decoder, '}'); err != nil {
		return nil, err
	}

	// Anything after the closing brace other than whitespace is an error.
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: data after the header object", ErrInvalidHeader)
	}
	return entries, nil
}

// expectDelimiter reads the next token and returns an error wrapping
// ErrInvalidHeader unless it is want.
func expectDelimiter(decoder *json.Decoder, want json.Delim) error {
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("%w: expecting %q: %w", ErrInvalidHeader, want, err)
	}
	if token != want {
		return fmt.Errorf("%w: expecting %q, found %v", ErrInvalidHeader, want, token)
	}
	return nil
}

// decodeMetadata decodes the metadata entry's value, which must be an object
// of strings (or null), returning an error wrapping ErrInvalidHeader
// otherwise.
func decodeMetadata(value json.RawMessage) (map[string]string, error) {
	var metadata map[string]string
	if err := json.Unmarshal(value, &metadata); err != nil {
		return nil, fmt.Errorf("%w: %s: %w", ErrInvalidHeader, metadataKey, err)
	}
	return metadata, nil
}

// decodeDescription decodes a tensor's header entry and checks that each
// field is present and well formed. It returns an error wrapping
// ErrInvalidHeader for a structural problem or ErrUnsupportedDType for an
// unknown dtype name.
func decodeDescription(name string, value json.RawMessage) (DType, []int, [2]uint64, error) {
	var description tensorDescription
	if err := json.Unmarshal(value, &description); err != nil {
		return 0, nil, [2]uint64{}, fmt.Errorf("%w: tensor %q: %w", ErrInvalidHeader, name, err)
	}
	if description.DType == nil || description.Shape == nil || description.DataOffsets == nil {
		return 0, nil, [2]uint64{}, fmt.Errorf("%w: tensor %q: needs dtype, shape, and data_offsets", ErrInvalidHeader, name)
	}
	dtype, ok := parseDType(*description.DType)
	if !ok {
		return 0, nil, [2]uint64{}, fmt.Errorf("%w: tensor %q: %q", ErrUnsupportedDType, name, *description.DType)
	}
	shape := *description.Shape
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, nil, [2]uint64{}, fmt.Errorf("%w: tensor %q: negative dimension in shape %v", ErrInvalidHeader, name, shape)
		}
	}
	offsets := *description.DataOffsets
	if len(offsets) != 2 {
		return 0, nil, [2]uint64{}, fmt.Errorf("%w: tensor %q: data_offsets has %d values, want 2", ErrInvalidHeader, name, len(offsets))
	}
	return dtype, shape, [2]uint64{offsets[0], offsets[1]}, nil
}
