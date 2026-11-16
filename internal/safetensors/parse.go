// Package safetensors parses the safetensors file format into tensor views
// that alias the parsed bytes. It validates every header before trusting it,
// so a malformed or hostile file yields an error instead of out-of-bounds
// views. The package never copies tensor payloads and uses no unsafe code.
package safetensors

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"math/bits"
	"slices"
)

// Layout constants of the safetensors format.
const (
	// lengthPrefixBytes is the size of the little-endian uint64 header length
	// that starts every file.
	lengthPrefixBytes = 8

	// maxHeaderBytes bounds the declared header length, matching the
	// reference implementation's limit, so a corrupt prefix cannot make Parse
	// decode an arbitrarily large header.
	maxHeaderBytes = 100 << 20
)

// span is one tensor's [begin, end) byte range within the data section.
type span struct {
	// name is the tensor owning the range, for error messages.
	name string

	// begin is the first byte of the range.
	begin uint64

	// end is one past the last byte of the range.
	end uint64
}

// Parse validates a safetensors file and returns its tensors and metadata.
//
// Each Tensor's Data aliases data, with its capacity limited to its length so
// appending to one tensor cannot overwrite another; data must therefore stay
// unmodified for as long as the returned File is in use. Parse checks the
// header's length, that it is a single JSON object without repeated names,
// that each entry names a known dtype, a non-negative shape, and two offsets,
// that each tensor's byte span equals its element count times its dtype
// size, and that the spans tile the data section exactly, with no overlap,
// gap, or trailing bytes.
//
// Errors wrap ErrTruncated, ErrInvalidHeader, ErrUnsupportedDType, or
// ErrInvalidOffsets. Parse is safe to call concurrently.
func Parse(data []byte) (File, error) {
	headerJSON, payload, err := splitFile(data)
	if err != nil {
		return File{}, err
	}
	entries, err := splitHeader(headerJSON)
	if err != nil {
		return File{}, err
	}

	// Decode each entry, checking each tensor against its own byte span.
	file := File{Tensors: make(map[string]Tensor, len(entries))}
	spans := make([]span, 0, len(entries))
	for _, entry := range entries {
		if entry.key == metadataKey {
			if file.Metadata, err = decodeMetadata(entry.value); err != nil {
				return File{}, err
			}
			continue
		}
		tensor, tensorSpan, err := decodeTensor(entry, uint64(len(payload)))
		if err != nil {
			return File{}, err
		}
		tensor.Data = payload[tensorSpan.begin:tensorSpan.end:tensorSpan.end]
		file.Tensors[entry.key] = tensor
		spans = append(spans, tensorSpan)
	}

	// Check the spans against one another.
	if err := checkTiling(spans, uint64(len(payload))); err != nil {
		return File{}, err
	}
	return file, nil
}

// splitFile separates data into its header JSON and its data section,
// returning an error wrapping ErrTruncated or ErrInvalidHeader when the length
// prefix is missing, too large, or points past the end of data.
func splitFile(data []byte) (headerJSON, payload []byte, err error) {
	if len(data) < lengthPrefixBytes {
		return nil, nil, fmt.Errorf("%w: %d bytes, need at least %d", ErrTruncated, len(data), lengthPrefixBytes)
	}
	headerLength := binary.LittleEndian.Uint64(data)
	if headerLength > maxHeaderBytes {
		return nil, nil, fmt.Errorf("%w: header length %d exceeds %d", ErrInvalidHeader, headerLength, maxHeaderBytes)
	}
	headerEnd := lengthPrefixBytes + int(headerLength)
	if headerEnd > len(data) {
		return nil, nil, fmt.Errorf("%w: header length %d, but only %d bytes follow", ErrTruncated, headerLength, len(data)-lengthPrefixBytes)
	}
	return data[lengthPrefixBytes:headerEnd], data[headerEnd:], nil
}

// decodeTensor decodes a tensor entry and checks that its offsets lie within
// a data section of payloadLength bytes and match its shape and dtype. The
// returned Tensor has no Data yet. Errors wrap ErrInvalidHeader,
// ErrUnsupportedDType, or ErrInvalidOffsets.
func decodeTensor(entry headerEntry, payloadLength uint64) (Tensor, span, error) {
	dtype, shape, offsets, err := decodeDescription(entry.key, entry.value)
	if err != nil {
		return Tensor{}, span{}, err
	}
	tensorSpan := span{name: entry.key, begin: offsets[0], end: offsets[1]}
	if tensorSpan.begin > tensorSpan.end || tensorSpan.end > payloadLength {
		return Tensor{}, span{}, fmt.Errorf("%w: tensor %q: offsets [%d, %d] outside a %d-byte data section",
			ErrInvalidOffsets, entry.key, tensorSpan.begin, tensorSpan.end, payloadLength)
	}
	byteCount, ok := tensorBytes(dtype, shape)
	if !ok || byteCount != tensorSpan.end-tensorSpan.begin {
		return Tensor{}, span{}, fmt.Errorf("%w: tensor %q: %v %v does not fill offsets [%d, %d]",
			ErrInvalidOffsets, entry.key, dtype, shape, tensorSpan.begin, tensorSpan.end)
	}
	return Tensor{Name: entry.key, DType: dtype, Shape: shape}, tensorSpan, nil
}

// tensorBytes returns the byte size of a tensor of the given dtype and
// shape, and false if the size overflows a uint64, dtype is not a known
// dtype, or shape holds a negative dimension.
func tensorBytes(dtype DType, shape []int) (uint64, bool) {
	elementSize := dtype.Size()
	if elementSize <= 0 {
		return 0, false
	}
	total := uint64(elementSize)
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, false
		}
		high, low := bits.Mul64(total, uint64(dimension))
		if high != 0 {
			return 0, false
		}
		total = low
	}
	return total, true
}

// checkTiling returns an error wrapping ErrInvalidOffsets unless spans,
// taken in offset order, cover [0, payloadLength) exactly once. Rejecting
// gaps and trailing bytes, as the format specification requires, keeps
// files from smuggling unindexed content. spans is not modified.
func checkTiling(spans []span, payloadLength uint64) error {
	ordered := slices.SortedFunc(slices.Values(spans), func(a, b span) int {
		return cmp.Or(cmp.Compare(a.begin, b.begin), cmp.Compare(a.end, b.end))
	})
	var position uint64
	for _, tensorSpan := range ordered {
		if tensorSpan.begin != position {
			return fmt.Errorf("%w: tensor %q starts at %d, want %d (overlap or gap)",
				ErrInvalidOffsets, tensorSpan.name, tensorSpan.begin, position)
		}
		position = tensorSpan.end
	}
	if position != payloadLength {
		return fmt.Errorf("%w: tensors end at %d, but the data section holds %d bytes",
			ErrInvalidOffsets, position, payloadLength)
	}
	return nil
}
