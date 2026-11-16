package chronos2

import (
	"encoding/binary"
	"unsafe"
)

// This file holds the library's only use of package unsafe. Checkpoints
// arrive as embedded byte slices, whose data the Go linker currently aligns
// (not a language guarantee; parity's alias test verifies it); viewing their
// weights in place as float32 values instead of copying them saves hundreds
// of megabytes per model. The view is read-only by convention, not in
// hardware: nothing in this module writes through a weight slice, and
// Checkpoint requires callers never to modify a shard after Load.

// float32Size is the width of one float32 in bytes.
const float32Size = 4

// nativeLittleEndian reports whether the platform stores multi-byte values
// little-endian, the byte order of safetensors data; only then may weight
// bytes be reinterpreted as float32 values in place.
var nativeLittleEndian = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

// aliasFloat32 returns data viewed as little-endian float32 values without
// copying, and true, when the platform is little-endian and data starts on a
// 4-byte boundary; otherwise it returns nil and false and the caller must
// decode a copy. len(data) must be a multiple of 4. The result shares data's
// memory and inherits its read-only contract.
func aliasFloat32(data []byte) ([]float32, bool) {
	if len(data) == 0 {
		return []float32{}, true
	}
	if !nativeLittleEndian || uintptr(unsafe.Pointer(&data[0]))%float32Size != 0 { // #nosec G103 -- alignment check only; the address is not dereferenced.
		return nil, false
	}
	// Alignment and byte order were checked above, and the length covers
	// exactly len(data) bytes, so every element is a valid, aligned float32
	// inside data.
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/float32Size), true // #nosec G103 -- aligned little-endian bytes checked above.
}
