package chronos2

import (
	"context"
	"slices"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
)

// forwardOptions controls one engine pass.
type forwardOptions struct {
	// parallelism is the maximum number of goroutines the pass may use, at
	// least 1.
	parallelism int

	// hidden requests the final hidden states alongside the quantiles.
	hidden bool
}

// forwarder runs one forward pass of the network over an engine batch. The
// root package depends on this interface rather than on *network.Network so
// that validation, batching, and result shaping are testable with a
// hand-written fake.
type forwarder interface {
	// Forward predicts batch.OutputPatches patches for every row of batch and,
	// when options.hidden is set, returns the final hidden states. It returns
	// an error matching ctx.Err() when ctx ends during the pass.
	Forward(ctx context.Context, batch network.Batch, options forwardOptions) (network.Output, error)
}

// longHorizonForecaster predicts horizon steps, beyond one pass, for every row
// of batch, returning Rows × Quantiles × horizon values in row, then
// quantile, then step order.
type longHorizonForecaster func(ctx context.Context, forward forwarder, batch batching.Batch, horizon, parallelism int) ([]float32, error)

// modelSpec describes the properties of a checkpoint the public API reports
// and relies on.
type modelSpec struct {
	// name is the checkpoint's name, reported by Model.Name.
	name string

	// contextLength is the longest history the model reads, in steps.
	contextLength int

	// outputPatchSize is the number of steps per output patch.
	outputPatchSize int

	// maxOutputPatches is the most output patches one pass predicts.
	maxOutputPatches int

	// quantiles lists the native quantile levels in (0, 1), ascending.
	quantiles []float64

	// embeddingDimension is the model width, the length of every embedding
	// vector.
	embeddingDimension int

	// regToken reports whether the encoder appends a [REG] token after the
	// context patches.
	regToken bool
}

// Model is a loaded Chronos-2 checkpoint, ready to forecast. Obtain one from
// a model package's New function, such as
// [github.com/pinkhop/chronos2-go-small.New], or from [Load]. A *Model
// is immutable and safe for concurrent use by any number of goroutines.
type Model struct {
	// spec describes the checkpoint.
	spec modelSpec

	// forward runs one engine pass.
	forward forwarder

	// unroll forecasts horizons beyond one pass.
	unroll longHorizonForecaster
}

// newModel assembles a Model from its parts; spec.quantiles is retained and
// must not be modified afterward.
func newModel(spec modelSpec, forward forwarder, unroll longHorizonForecaster) *Model {
	return &Model{spec: spec, forward: forward, unroll: unroll}
}

// Name returns the checkpoint's name, such as "chronos-2" or
// "chronos-2-small".
func (m *Model) Name() string { return m.spec.name }

// ContextLength returns the longest history the model reads, in steps; a
// longer history is trimmed to its most recent ContextLength steps. It is 8192
// for both published models, and it is the upper bound for
// [WithContextLength].
func (m *Model) ContextLength() int { return m.spec.contextLength }

// MaxHorizon returns the longest horizon one pass of the model predicts, in
// steps; longer horizons need [WithLongHorizon]. It is 1024 for both
// published models.
func (m *Model) MaxHorizon() int { return m.spec.outputPatchSize * m.spec.maxOutputPatches }

// NativePercentiles returns, in ascending order, the percentile levels the
// model predicts directly: 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95,
// and 99 for the small model, and those plus 15, 25, 35, 45, 55, 65, 75,
// and 85 for the base model. [Forecast.Percentile] interpolates any other
// level. The caller owns the returned slice.
func (m *Model) NativePercentiles() []float64 { return levelsToPercents(m.spec.quantiles) }

// EmbeddingDimension returns the number of elements in every [Embedding]
// vector: 768 for the base model and 512 for the small model.
func (m *Model) EmbeddingDimension() int { return m.spec.embeddingDimension }

// percentScale converts a quantile level in (0, 1) to a percent.
const percentScale = 100

// percentRounding is the number of decimal places a reported percent keeps,
// hiding binary rounding such as 0.07 × 100 = 7.000000000000001.
const percentRounding = 1e9

// levelsToPercents converts quantile levels to percents, rounded to hide
// binary representation error, in a new slice.
func levelsToPercents(levels []float64) []float64 {
	percents := slices.Clone(levels)
	for i, level := range percents {
		percents[i] = roundPercent(level * percentScale)
	}
	return percents
}
