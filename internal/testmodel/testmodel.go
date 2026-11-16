// Package testmodel gives tests access to the synthetic Chronos-2 checkpoint
// and to the reference fixtures generated from chronos-forecasting 2.3.2. It
// is imported only by tests.
package testmodel

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// SyntheticDir returns the absolute path of the root module's
// testdata/synthetic directory, which holds the synthetic checkpoint (a tiny
// randomly initialized Chronos-2: two layers, width 32) and its stage
// fixtures. The scenario fixtures are in its scenarios subdirectory.
func SyntheticDir() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		panic("testmodel: cannot locate source file")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "testdata", "synthetic")
}

// SyntheticCheckpoint returns the synthetic checkpoint's config.json and its
// single safetensors shard, ready for chronos2.Checkpoint. It fails the test
// when the files cannot be read.
func SyntheticCheckpoint(tb testing.TB) (config []byte, shards [][]byte) {
	tb.Helper()
	dir := SyntheticDir()
	// #nosec G304 -- tests pass fixed repository fixture paths; no untrusted input.
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		tb.Fatalf("read synthetic config: %v", err)
	}
	// #nosec G304 -- tests pass fixed repository fixture paths; no untrusted input.
	shard, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		tb.Fatalf("read synthetic weights: %v", err)
	}
	return config, [][]byte{shard}
}
