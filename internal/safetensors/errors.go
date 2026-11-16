package safetensors

import "errors"

// Sentinel errors. Every error this package returns wraps exactly one of them
// with context, so callers classify failures with errors.Is.
var (
	// ErrTruncated reports input too short to hold its length prefix, or its
	// declared header.
	ErrTruncated = errors.New("safetensors: truncated data")

	// ErrInvalidHeader reports a header that is too long, is not a single JSON
	// object, repeats a tensor name, or holds an entry that is not a
	// well-formed tensor description or string-valued metadata.
	ErrInvalidHeader = errors.New("safetensors: invalid header")

	// ErrUnsupportedDType reports a tensor whose dtype is not a safetensors
	// dtype name.
	ErrUnsupportedDType = errors.New("safetensors: unsupported dtype")

	// ErrInvalidOffsets reports tensor offsets that disagree with the tensor's
	// shape and dtype, fall outside the data section, overlap, or leave bytes
	// of the data section unclaimed.
	ErrInvalidOffsets = errors.New("safetensors: invalid tensor offsets")

	// ErrDTypeMismatch reports a typed accessor called on a tensor of another
	// dtype.
	ErrDTypeMismatch = errors.New("safetensors: dtype mismatch")
)
