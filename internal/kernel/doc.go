// Package kernel holds the numeric primitives that every Chronos-2 forward
// pass is built from: matrix products, normalization, softmax, activations,
// residual sums, and rotary position embeddings.
//
// Every function follows one contract so that a SIMD implementation can
// replace it without touching callers. Each operates on flat, row-major
// []float32 slices with explicit dimensions, writes into caller-provided
// output slices, never allocates, and never starts goroutines; parallelism
// belongs to the caller. Slice lengths must equal the sizes the dimensions
// imply, and a mismatch panics, because it is always a programming error in
// the caller. Kernels never return errors.
//
// Each operation lives in its own file with a _generic.go suffix, leaving
// room for a sibling _simd.go implementation selected by a build constraint.
package kernel
