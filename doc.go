// Package chronos2 forecasts time series with Amazon's Chronos-2 model,
// running entirely in-process with the model's weights embedded in the
// binary. It needs no network access, no files, and no cgo.
//
// Chronos-2 is a pretrained forecasting model: it was trained once, by
// Amazon, on a large collection of real and synthetic series, and forecasts
// new series without being trained on them. Give it a history (a series of
// observations taken at a fixed interval, such as hourly or daily, oldest
// first) and a horizon (how many future steps to predict, at the same
// interval), and it returns a probabilistic forecast: for each future step,
// a range of plausible values rather than a single guess. It can also
// forecast several related series together and use covariates, series that
// inform a forecast without being forecast themselves.
//
// # Obtaining a Model
//
// Each published checkpoint (a saved copy of the trained model) has its own
// package and Go module, which embeds the checkpoint's weights in any binary
// that imports it:
//
//   - [github.com/pinkhop/chronos2-go-small] embeds
//     autogluon/chronos-2-small, about 112 MB.
//   - [github.com/pinkhop/chronos2-go-base] embeds amazon/chronos-2,
//     about 478 MB. It is slower and predicts more percentile levels
//     natively.
//
// Both expose the same constructor, which decodes the checkpoint on its first
// call and returns the same shared [*Model] afterward:
//
//	m, err := small.New()
//	if err != nil {
//		return fmt.Errorf("load forecasting model: %w", err)
//	}
//
// Switching models means changing one import path. [Load] builds a Model from
// any other Chronos-2 checkpoint, such as a fine-tuned one.
//
// # Forecasting One Series
//
// The univariate path is a single call:
//
//	fc, err := m.Forecast(ctx, history, 48)
//
// A history may hold up to [Model.ContextLength] steps (8192 for both
// published models); a longer history is trimmed to its most recent steps.
// The horizon may be up to [Model.MaxHorizon] steps (1024 for both models)
// in one pass of the model; see Long Horizons for more.
//
// # Reading Results
//
// A [Forecast] describes the predicted distribution through percentiles: the
// p-th percentile line holds, for each future step, the value the model
// expects the actual value to fall below with probability p%. The 50th
// percentile, the median, is the usual point forecast; the 10th and 90th
// percentiles bound an 80% prediction interval.
//
//	median := fc.Median()            // []float64, one value per future step
//	p90 := fc.Percentile90()         // predicted natively by both models
//	p15 := fc.Percentile15()         // nil on the small model
//	p925, err := fc.Percentile(92.5) // interpolated between native levels
//
// Each model predicts a fixed set of native percentiles directly, reported by
// [Model.NativePercentiles]: 13 levels for the small model and 21 for the
// base model. The named accessors [Forecast.Percentile1] through
// [Forecast.Percentile99] cover the base model's levels and return nil when
// the producing model does not predict a level natively; a native line is
// never empty, so nil always means "not native". [Forecast.Percentile]
// accepts any percent from 1 to 99 on either model, interpolating linearly
// between native levels. Every accessor returns a new slice the caller may
// modify.
//
// # Forecasting Many Series
//
// The other methods extend the univariate call:
//
//   - [Model.ForecastBatch] forecasts many independent histories, which may
//     differ in length, in one call.
//   - [Model.ForecastMultivariate] forecasts variates, several co-evolving
//     series measured on one time axis, such as the temperature, humidity,
//     and pressure at one weather station. Each variate's forecast draws on
//     the others.
//   - [Model.ForecastWithCovariates] forecasts one target, the series to
//     predict, informed by covariates. A past-only covariate is known only
//     historically, such as yesterday's foot traffic; a known-future
//     covariate also has values over the horizon, such as a weather forecast
//     or a promotion calendar. Covariates may be numeric or categorical
//     (string labels such as "holiday" or "weekday"); see [Covariates].
//   - [Model.ForecastTasks] is the general form. A [Task] is a set of series
//     the model reads together: one or more targets plus their covariates,
//     all on one time axis. Tasks are independent unless the call passes
//     [WithCrossLearning].
//
// The first three convert their input into tasks and follow the same path
// as ForecastTasks.
//
// # Embeddings
//
// [Model.Embed] returns an [Embedding]: the vectors the model builds
// internally while reading a history, arranged so that series that behave
// alike get similar vectors. [Embedding.Summary] gives one vector per
// history for similarity search, clustering, or features for other models.
// The model normalizes each history by its mean and standard deviation
// first, so embeddings describe shape, not level; [Embedding.Location] and
// [Embedding.Scale] report the level and spread that were removed.
//
// # Options
//
// Options are passed variadically after the required arguments:
//
//	fc, err := m.Forecast(ctx, history, 48, chronos2.WithContextLength(512))
//
// [WithContextLength] limits how much history the model reads, which runs
// faster. [WithMaxParallelism] caps the goroutines one call uses.
// [WithBatchSize] bounds how many series one pass processes together, which
// bounds memory. [WithLongHorizon] permits horizons beyond MaxHorizon, and
// [WithCrossLearning] lets the tasks of one call share information. An
// out-of-range option value makes the call return [ErrInvalidOption].
//
// # Missing Values
//
// NaN marks a missing numeric observation in a history, target, or numeric
// covariate; the model ignores it. A history needs at least one observed
// value. Infinity is rejected with [ErrInvalidValue], as is any finite value
// too large for float32, the precision the model computes in (magnitude
// beyond about 3.4 × 10³⁸). In a categorical covariate, the empty string
// marks a missing label; the model treats it as a label of its own rather
// than ignoring it.
//
// # Errors
//
// Every error a method returns matches exactly one of the package's sentinel
// errors under [errors.Is], such as [ErrEmptyHistory], [ErrMismatchedLengths],
// or [ErrHorizonTooLong], and wraps it with context such as the offending
// task, target, or covariate. A call whose context is canceled or expires
// returns an error matching the context's error instead.
//
// # Concurrency
//
// A [*Model] is immutable and safe for concurrent use by any number of
// goroutines, as are [Forecast] and [Embedding] values. One call spreads its
// arithmetic across up to GOMAXPROCS goroutines, all finished before the call
// returns. A server handling many requests at once should pass
// WithMaxParallelism(1) and let its request concurrency use the cores
// instead.
//
// # Long Horizons
//
// One pass of the model predicts at most [Model.MaxHorizon] steps. With
// [WithLongHorizon], longer horizons are reached by feeding the model's own
// forecasts back to it as history, in rounds of up to MaxHorizon steps. Each
// extra round costs about nine passes, and forecasts far beyond MaxHorizon
// rest on predicted rather than observed values, so their uncertainty is
// harder to trust.
package chronos2
