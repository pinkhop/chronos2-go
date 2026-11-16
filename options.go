package chronos2

import (
	"fmt"
	"runtime"
)

// defaultBatchSize is the number of series rows per engine pass when the
// caller passes no WithBatchSize option.
const defaultBatchSize = 256

// ForecastOption configures a forecasting method such as [Model.Forecast].
// Only the With functions in this package create options; other packages
// cannot implement the interface.
type ForecastOption interface {
	// applyForecast records the option in settings.
	applyForecast(settings *callSettings)
}

// EmbedOption configures [Model.Embed]. Only [WithContextLength] and
// [WithMaxParallelism] create embed options; other packages cannot implement
// the interface.
type EmbedOption interface {
	// applyEmbed records the option in settings.
	applyEmbed(settings *callSettings)
}

// SharedOption is an option accepted by both the forecasting methods and
// [Model.Embed], returned by [WithContextLength] and [WithMaxParallelism].
// The zero SharedOption has no effect.
type SharedOption struct {
	// apply records the option in a call's settings.
	apply func(settings *callSettings)
}

// applyForecast records the option for a forecasting call.
func (o SharedOption) applyForecast(settings *callSettings) { o.applyShared(settings) }

// applyEmbed records the option for an Embed call.
func (o SharedOption) applyEmbed(settings *callSettings) { o.applyShared(settings) }

// applyShared records the option, tolerating the zero SharedOption, which
// has no effect.
func (o SharedOption) applyShared(settings *callSettings) {
	if o.apply != nil {
		o.apply(settings)
	}
}

// forecastOnlyOption is an option accepted only by the forecasting methods.
type forecastOnlyOption func(settings *callSettings)

// applyForecast records the option for a forecasting call.
func (o forecastOnlyOption) applyForecast(settings *callSettings) { o(settings) }

// callSettings collects one call's options. Zero values mean "absent";
// resolve substitutes defaults and checks ranges against the model.
type callSettings struct {
	// contextLength is the WithContextLength value, meaningful only when contextLengthSet.
	contextLength int

	// contextLengthSet reports whether the call passed WithContextLength.
	contextLengthSet bool

	// maxParallelism is the WithMaxParallelism value, meaningful only when maxParallelismSet.
	maxParallelism int

	// maxParallelismSet reports whether the call passed WithMaxParallelism.
	maxParallelismSet bool

	// batchSize is the WithBatchSize value, meaningful only when batchSizeSet.
	batchSize int

	// batchSizeSet reports whether the call passed WithBatchSize.
	batchSizeSet bool

	// longHorizon reports whether WithLongHorizon was passed.
	longHorizon bool

	// crossLearning reports whether WithCrossLearning was passed.
	crossLearning bool
}

// WithContextLength makes a call read at most the most recent steps
// observations of each series, ignoring older ones. Shorter windows run
// faster but give the model less history to learn patterns from. Valid values
// run from 1 to [Model.ContextLength], the default; any other value makes the
// call return [ErrInvalidOption].
func WithContextLength(steps int) SharedOption {
	return SharedOption{apply: func(settings *callSettings) {
		settings.contextLength, settings.contextLengthSet = steps, true
	}}
}

// WithMaxParallelism caps the goroutines one call uses for its arithmetic; 1
// runs the call on the calling goroutine only. The default is GOMAXPROCS.
// Results are identical for every value. A server running many calls at once
// should pass 1 and parallelize across calls instead. A value below 1 makes
// the call return [ErrInvalidOption].
func WithMaxParallelism(goroutines int) SharedOption {
	return SharedOption{apply: func(settings *callSettings) {
		settings.maxParallelism, settings.maxParallelismSet = goroutines, true
	}}
}

// WithBatchSize caps how many series the model processes together in one
// pass, counting every target and every covariate as one series; the default
// is 256. Smaller batches use less memory. A call holding more series runs
// in several passes, keeping each task's series together, so a task larger
// than the cap still runs in one pass of its own. A value below 1 makes the
// call return [ErrInvalidOption].
func WithBatchSize(rows int) ForecastOption {
	return forecastOnlyOption(func(settings *callSettings) {
		settings.batchSize, settings.batchSizeSet = rows, true
	})
}

// WithLongHorizon permits horizons beyond [Model.MaxHorizon], the most steps
// one pass of the model predicts. Longer horizons are reached in rounds that
// feed the model's own forecasts back to it as history; each extra round of
// up to MaxHorizon steps costs about nine passes, and the far steps rest on
// predicted rather than observed values. Without this option, such a horizon
// returns [ErrHorizonTooLong].
func WithLongHorizon() ForecastOption {
	return forecastOnlyOption(func(settings *callSettings) { settings.longHorizon = true })
}

// WithCrossLearning lets every task in the call share information, as if all
// of its series formed one task, which can help when many short, related
// series each carry little history. The whole call must then fit one pass of
// at most [WithBatchSize] series, or the call returns
// [ErrCrossLearningBatchTooLarge]. It has no effect on a call with a single
// task.
func WithCrossLearning() ForecastOption {
	return forecastOnlyOption(func(settings *callSettings) { settings.crossLearning = true })
}

// resolvedSettings holds a call's validated options with defaults applied.
type resolvedSettings struct {
	// contextLength is the trim window, from 1 to the model's context length.
	contextLength int

	// parallelism is the maximum number of goroutines, at least 1.
	parallelism int

	// batchSize is the maximum rows per engine pass, at least 1.
	batchSize int

	// longHorizon permits horizons beyond the single-pass maximum.
	longHorizon bool

	// crossLearning puts every series of the call in one group.
	crossLearning bool
}

// resolve applies defaults to settings and checks each present option
// against a model whose context length is maxContextLength, returning an
// error wrapping ErrInvalidOption for an out-of-range value.
func (settings callSettings) resolve(maxContextLength int) (resolvedSettings, error) {
	resolved := resolvedSettings{
		contextLength: maxContextLength,
		parallelism:   runtime.GOMAXPROCS(0),
		batchSize:     defaultBatchSize,
		longHorizon:   settings.longHorizon,
		crossLearning: settings.crossLearning,
	}
	if settings.contextLengthSet {
		if settings.contextLength < 1 || settings.contextLength > maxContextLength {
			return resolvedSettings{}, fmt.Errorf("%w: context length %d outside [1, %d]", ErrInvalidOption, settings.contextLength, maxContextLength)
		}
		resolved.contextLength = settings.contextLength
	}
	if settings.maxParallelismSet {
		if settings.maxParallelism < 1 {
			return resolvedSettings{}, fmt.Errorf("%w: max parallelism %d below 1", ErrInvalidOption, settings.maxParallelism)
		}
		resolved.parallelism = settings.maxParallelism
	}
	if settings.batchSizeSet {
		if settings.batchSize < 1 {
			return resolvedSettings{}, fmt.Errorf("%w: batch size %d below 1", ErrInvalidOption, settings.batchSize)
		}
		resolved.batchSize = settings.batchSize
	}
	return resolved, nil
}

// collectForecastOptions applies options in order, later options overriding
// earlier ones; nil options are ignored.
func collectForecastOptions(options []ForecastOption) callSettings {
	var settings callSettings
	for _, option := range options {
		if option != nil {
			option.applyForecast(&settings)
		}
	}
	return settings
}

// collectEmbedOptions applies options in order, later options overriding
// earlier ones; nil options are ignored.
func collectEmbedOptions(options []EmbedOption) callSettings {
	var settings callSettings
	for _, option := range options {
		if option != nil {
			option.applyEmbed(&settings)
		}
	}
	return settings
}
