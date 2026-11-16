package safetensors

import "strconv"

// DType identifies the element type of a tensor as the safetensors header
// names it. Every safetensors dtype parses, but only F32 has decoding support
// in this package.
type DType int

// The dtypes a safetensors header may declare.
const (
	// DTypeUnspecified is the zero value; Parse never returns it.
	DTypeUnspecified DType = iota

	// Bool is a one-byte boolean ("BOOL").
	Bool

	// U8 is an unsigned 8-bit integer ("U8").
	U8

	// I8 is a signed 8-bit integer ("I8").
	I8

	// F8E5M2 is an 8-bit float with 5 exponent and 2 mantissa bits ("F8_E5M2").
	F8E5M2

	// F8E4M3 is an 8-bit float with 4 exponent and 3 mantissa bits ("F8_E4M3").
	F8E4M3

	// I16 is a signed 16-bit integer ("I16").
	I16

	// U16 is an unsigned 16-bit integer ("U16").
	U16

	// F16 is an IEEE 754 half-precision float ("F16").
	F16

	// BF16 is a bfloat16 ("BF16").
	BF16

	// I32 is a signed 32-bit integer ("I32").
	I32

	// U32 is an unsigned 32-bit integer ("U32").
	U32

	// F32 is an IEEE 754 single-precision float ("F32").
	F32

	// F64 is an IEEE 754 double-precision float ("F64").
	F64

	// I64 is a signed 64-bit integer ("I64").
	I64

	// U64 is an unsigned 64-bit integer ("U64").
	U64
)

// dtypeInfo pairs a dtype's header name with its element width.
type dtypeInfo struct {
	// name is the dtype's spelling in a safetensors header.
	name string

	// size is the width of one element in bytes.
	size int
}

// dtypeInfos describes every DType except DTypeUnspecified, indexed by DType.
var dtypeInfos = [...]dtypeInfo{
	Bool:   {"BOOL", 1},
	U8:     {"U8", 1},
	I8:     {"I8", 1},
	F8E5M2: {"F8_E5M2", 1},
	F8E4M3: {"F8_E4M3", 1},
	I16:    {"I16", 2},
	U16:    {"U16", 2},
	F16:    {"F16", 2},
	BF16:   {"BF16", 2},
	I32:    {"I32", 4},
	U32:    {"U32", 4},
	F32:    {"F32", 4},
	F64:    {"F64", 8},
	I64:    {"I64", 8},
	U64:    {"U64", 8},
}

// known reports whether d is a DType this package defines, other than
// DTypeUnspecified.
func (d DType) known() bool {
	return d > DTypeUnspecified && int(d) < len(dtypeInfos)
}

// Size returns the width of one element of d in bytes, or 0 for
// DTypeUnspecified and values outside the defined set.
func (d DType) Size() int {
	if !d.known() {
		return 0
	}
	return dtypeInfos[d].size
}

// String returns d's safetensors header name, such as "F32", so errors and
// test output read like the file format. DTypeUnspecified and undefined
// values render as Go identifiers instead.
func (d DType) String() string {
	switch {
	case d == DTypeUnspecified:
		return "DTypeUnspecified"
	case d.known():
		return dtypeInfos[d].name
	default:
		return "DType(" + strconv.Itoa(int(d)) + ")"
	}
}

// parseDType returns the DType a header names, matching case-sensitively as
// the format requires, and false when name is not a safetensors dtype.
func parseDType(name string) (DType, bool) {
	for d := Bool; int(d) < len(dtypeInfos); d++ {
		if dtypeInfos[d].name == name {
			return d, true
		}
	}
	return DTypeUnspecified, false
}
