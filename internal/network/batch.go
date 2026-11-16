package network

// Batch is the engine batch: the boundary between preprocessing and the
// forward pass. It holds Rows series, each described only by its values, its
// group, and its known future values; the network knows nothing about tasks,
// targets, or covariates.
//
// A Batch is read-only to the network, which never modifies or retains its
// slices.
type Batch struct {
	// Rows is the number of series in the batch, R. It is at least 1.
	Rows int

	// ContextLength is the number of steps per row in Context, L. It is at
	// least 1 and at most the model's context length.
	ContextLength int

	// Context holds Rows × ContextLength values, row-major, oldest step first.
	// NaN marks a missing value and the left padding of rows shorter than
	// ContextLength.
	Context []float32

	// GroupIDs holds one group per row. Rows sharing an ID attend to one
	// another in group attention; IDs need not be contiguous or sorted.
	GroupIDs []int

	// OutputPatches is the number of 16-step patches to predict, P, from 1 to
	// the model's maximum output patches.
	OutputPatches int

	// FutureCovariates holds Rows × (16 × OutputPatches) values, row-major. A
	// finite value is a known future value of a covariate row; NaN marks an
	// unknown value, and every value of a target or past-only covariate row is
	// NaN.
	FutureCovariates []float32
}

// Output is the result of one forward pass over a Batch.
type Output struct {
	// Rows is the number of rows, matching the Batch.
	Rows int

	// Quantiles is the number of quantile levels the model predicts, Q.
	Quantiles int

	// Steps is the number of predicted steps per row, 16 × OutputPatches.
	Steps int

	// Values holds Rows × Quantiles × Steps predictions in the caller's units
	// (denormalized), row-major: row, then quantile level, then step.
	Values []float32

	// Locations holds each row's normalization mean.
	Locations []float32

	// Scales holds each row's normalization standard deviation.
	Scales []float32

	// Tokens is the number of encoder positions per row, S: context patches,
	// the [REG] token, and the output patches. It is set only when hidden
	// states were requested.
	Tokens int

	// Width is the model width, D. It is set only when hidden states were
	// requested.
	Width int

	// Hidden holds Rows × Tokens × Width final hidden states (after the final
	// norm) when requested, and is nil otherwise.
	Hidden []float32
}
