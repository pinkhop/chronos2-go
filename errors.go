package chronos2

import "errors"

// Sentinel errors. Every error a [Model] method or [Load] returns wraps exactly
// one of them, or the context's error when a call is canceled or times out,
// with details such as the offending task, target, or covariate name. Classify
// failures with [errors.Is]; the message text is not stable.
var (
	// ErrEmptyHistory reports a history or target with no elements, or a task
	// with no targets.
	ErrEmptyHistory = errors.New("chronos2: empty history")

	// ErrNoObservations reports a history or target whose every value is NaN.
	ErrNoObservations = errors.New("chronos2: history has no observed values")

	// ErrInvalidValue reports an input holding positive or negative infinity,
	// or a finite value too large to represent in float32 (magnitude beyond
	// math.MaxFloat32, about 3.4 × 10³⁸), the precision the model computes in.
	ErrInvalidValue = errors.New("chronos2: input value is infinite or beyond the float32 range")

	// ErrInvalidHorizon reports a horizon below 1.
	ErrInvalidHorizon = errors.New("chronos2: horizon must be at least 1")

	// ErrHorizonTooLong reports a horizon beyond [Model.MaxHorizon] in a call
	// without [WithLongHorizon].
	ErrHorizonTooLong = errors.New("chronos2: horizon exceeds the model's single-pass maximum")

	// ErrMismatchedLengths reports series within one task that differ in
	// length, or a future covariate whose length differs from the horizon.
	ErrMismatchedLengths = errors.New("chronos2: mismatched series lengths")

	// ErrInvalidCovariates reports covariate names that break the structural
	// rules: a future key without a past key, or a name in both the numeric
	// and categorical maps.
	ErrInvalidCovariates = errors.New("chronos2: invalid covariates")

	// ErrInvalidOption reports an option carrying an out-of-range value, such
	// as WithBatchSize(0) or a WithContextLength beyond
	// [Model.ContextLength].
	ErrInvalidOption = errors.New("chronos2: invalid option")

	// ErrCrossLearningBatchTooLarge reports a call with [WithCrossLearning]
	// whose series (every target and every covariate of every task) outnumber
	// the batch size set by [WithBatchSize], 256 by default. Cross-learning
	// must process the whole call in one pass of the model.
	ErrCrossLearningBatchTooLarge = errors.New("chronos2: cross-learning call exceeds the batch size")

	// ErrPercentileOutOfRange reports a requested percentile outside [1, 99].
	ErrPercentileOutOfRange = errors.New("chronos2: percentile outside [1, 99]")

	// ErrInvalidCheckpoint reports a checkpoint that failed to decode or
	// validate in [Load], or, from a forecasting method, a model whose output
	// disagrees with its configuration. With the published model packages,
	// it indicates damaged embedded files or a bug in this module.
	ErrInvalidCheckpoint = errors.New("chronos2: invalid checkpoint")
)
