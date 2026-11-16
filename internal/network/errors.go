package network

import "errors"

// Sentinel errors returned by this package, wrapped with context. Callers
// compare them with errors.Is.
var (
	// ErrInvalidConfig reports a config.json that is malformed or describes
	// an architecture this engine does not support.
	ErrInvalidConfig = errors.New("network: invalid model config")

	// ErrInvalidWeights reports a checkpoint tensor that is missing or whose
	// shape disagrees with the config.
	ErrInvalidWeights = errors.New("network: invalid model weights")

	// ErrInvalidBatch reports an engine batch whose dimensions or slice
	// lengths are inconsistent or exceed the model's limits.
	ErrInvalidBatch = errors.New("network: invalid batch")
)
