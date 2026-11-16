package horizon

import (
	"context"
	"math"

	"github.com/pinkhop/chronos2-go/internal/batching"
	"github.com/pinkhop/chronos2-go/internal/network"
	"github.com/pinkhop/chronos2-go/internal/quantile"
)

// unrolledLevels are the deciles whose paths long-horizon rounds follow,
// matching the reference's default unrolled_quantiles. It is an array so its
// length is a constant; callers pass unrolledLevels[:] and never modify it.
var unrolledLevels = [...]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

// pathCount is the number of paths each original row expands into, one per
// unrolled decile.
const pathCount = len(unrolledLevels)

// paths holds the expanded rows of an unroll: row b's path n is expanded row
// b × pathCount + n, matching the reference's (b n) flattening.
type paths struct {
	// contextLength is the number of steps per expanded row.
	contextLength int

	// context holds rows × pathCount × contextLength values.
	context []float32

	// groupIDs holds group × pathCount + path for each expanded row, so a
	// path shares information only with the same path of its own task.
	groupIDs []int
}

// unroll predicts the steps after the first pass's prediction, writing each
// round's reduced forecast into result, until horizon steps are covered.
func unroll(ctx context.Context, f Forwarder, batch batching.Batch, horizon int, config Config, prediction pass, result []float32) error {
	weights := sampleWeights(config.Levels)
	state := expandPaths(batch)
	done := prediction.steps
	for number := 2; done < horizon; number++ {
		// Extend each path with its decile of the previous prediction, keeping
		// known future covariate values where the batch supplies them.
		state = state.extend(prediction, batch, config, done-prediction.steps)

		patches := outputPatches(horizon-done, config.MaxOutputPatches)
		expanded := network.Batch{
			Rows:             batch.Rows * pathCount,
			ContextLength:    state.contextLength,
			Context:          state.context,
			GroupIDs:         state.groupIDs,
			OutputPatches:    patches,
			FutureCovariates: futureWindow(batch, pathCount, done, patches*outputPatchSize),
		}
		output, err := forward(ctx, f, expanded, len(config.Levels), number)
		if err != nil {
			return err
		}

		prediction = reduce(output, batch.Rows, config.Levels, weights)
		place(result, prediction, done, horizon, batch.Rows, len(config.Levels))
		done += prediction.steps
	}
	return nil
}

// sampleWeights returns the probability mass of each (path, level) sample,
// flattened path-major: the outer product of the deciles' and the native
// levels' masses, each product computed in float32 as the reference does.
func sampleWeights(levels []float64) []float64 {
	pathMass, levelMass := quantile.LevelMass(unrolledLevels[:]), quantile.LevelMass(levels)
	weights := make([]float64, 0, len(pathMass)*len(levelMass))
	for _, p := range pathMass {
		for _, l := range levelMass {
			weights = append(weights, float64(float32(p)*float32(l)))
		}
	}
	return weights
}

// expandPaths repeats each of batch's rows pathCount times and shifts the
// group IDs so paths mix only with matching paths of the same group.
func expandPaths(batch batching.Batch) paths {
	length := batch.ContextLength
	expanded := paths{
		contextLength: length,
		context:       make([]float32, batch.Rows*pathCount*length),
		groupIDs:      make([]int, batch.Rows*pathCount),
	}
	for b := range batch.Rows {
		source := batch.Context[b*length : (b+1)*length]
		for n := range pathCount {
			r := b*pathCount + n
			copy(expanded.context[r*length:], source)
			expanded.groupIDs[r] = batch.GroupIDs[b]*pathCount + n
		}
	}
	return expanded
}

// extend returns new paths whose contexts append each path's decile of
// prediction, keep the most recent config.ContextLength steps, and then take
// batch's known future values over the appended steps, which begin at future
// step slideStart. The receiver is not modified.
func (p paths) extend(prediction pass, batch batching.Batch, config Config, slideStart int) paths {
	appended := prediction.steps
	length := min(p.contextLength+appended, config.ContextLength)
	extended := paths{contextLength: length, context: make([]float32, len(p.groupIDs)*length), groupIDs: p.groupIDs}
	quantiles := len(config.Levels)
	for b := range batch.Rows {
		lines := make([][]float32, quantiles)
		for q := range lines {
			offset := (b*quantiles + q) * appended
			lines[q] = prediction.values[offset : offset+appended]
		}
		deciles := quantile.Interpolate(unrolledLevels[:], config.Levels, lines)
		for n, decile := range deciles {
			r := b*pathCount + n
			row := extended.context[r*length : (r+1)*length]
			appendTrimmed(row, p.context[r*p.contextLength:(r+1)*p.contextLength], decile)
			slideKnownFuture(row[length-appended:], batch, b, slideStart)
		}
	}
	return extended
}

// appendTrimmed fills row with the last len(row) values of old followed by
// tail; len(row) is at least len(tail).
func appendTrimmed(row, old, tail []float32) {
	skip := len(old) + len(tail) - len(row)
	kept := copy(row, old[skip:])
	copy(row[kept:], tail)
}

// slideKnownFuture overwrites each value of tail with row b's future value at
// the matching step from start, wherever that value is known (not NaN).
func slideKnownFuture(tail []float32, batch batching.Batch, b, start int) {
	for i := range tail {
		step := start + i
		if step >= batch.FutureLength {
			return
		}
		if value := batch.Future[b*batch.FutureLength+step]; !math.IsNaN(float64(value)) {
			tail[i] = value
		}
	}
}

// reduce collapses an expanded pass's pathCount × Q predictions per original
// row and step into Q values at levels, using the weighted quantile of the
// samples ordered path-major to match weights.
func reduce(output pass, rows int, levels, weights []float64) pass {
	quantiles, steps := len(levels), output.steps
	values := make([]float32, rows*quantiles*steps)
	samples := make([]float32, pathCount*quantiles)
	for b := range rows {
		for step := range steps {
			gatherSamples(samples, output, quantiles, b, step)
			for q, value := range quantile.WeightedQuantile(levels, weights, samples) {
				values[(b*quantiles+q)*steps+step] = value
			}
		}
	}
	return pass{values: values, steps: steps}
}

// gatherSamples fills samples with every path's quantiles predictions for
// original row b at step, path-major.
func gatherSamples(samples []float32, output pass, quantiles, b, step int) {
	steps := output.steps
	for n := range pathCount {
		row := b*pathCount + n
		for q := range quantiles {
			samples[n*quantiles+q] = output.values[(row*quantiles+q)*steps+step]
		}
	}
}

// nanFloats returns n NaN values.
func nanFloats(n int) []float32 {
	values := make([]float32, n)
	nan := float32(math.NaN())
	for i := range values {
		values[i] = nan
	}
	return values
}
