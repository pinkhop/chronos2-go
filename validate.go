package chronos2

import (
	"fmt"
	"maps"
	"math"
	"slices"
)

// targetLabeler names a task's target in error messages, so each convenience
// method can describe its input in its own terms ("history 3", "variate 1").
type targetLabeler func(task, target int) string

// labelTaskTarget names targets by task and target index, for ForecastTasks.
func labelTaskTarget(task, target int) string {
	return fmt.Sprintf("task %d target %d", task, target)
}

// checkHorizon returns an error wrapping ErrInvalidHorizon for a horizon
// below 1, or ErrHorizonTooLong for one beyond maxHorizon unless longHorizon
// permits it.
func checkHorizon(horizon, maxHorizon int, longHorizon bool) error {
	if horizon < 1 {
		return fmt.Errorf("%w: got %d", ErrInvalidHorizon, horizon)
	}
	if horizon > maxHorizon && !longHorizon {
		return fmt.Errorf("%w: %d steps exceed %d; pass WithLongHorizon to permit it", ErrHorizonTooLong, horizon, maxHorizon)
	}
	return nil
}

// validateTasks checks every task against the input rules for a forecast of
// horizon steps, returning the first violation wrapped with its location.
func validateTasks(tasks []Task, horizon int, label targetLabeler) error {
	for taskIndex, task := range tasks {
		if len(task.Targets) == 0 {
			return fmt.Errorf("%w: task %d has no targets", ErrEmptyHistory, taskIndex)
		}
		length := len(task.Targets[0])
		for targetIndex, target := range task.Targets {
			if err := checkHistory(target, length); err != nil {
				return fmt.Errorf("%s: %w", label(taskIndex, targetIndex), err)
			}
		}
		if err := validateCovariates(task.Covariates, length, horizon); err != nil {
			return fmt.Errorf("task %d: %w", taskIndex, err)
		}
	}
	return nil
}

// checkHistory checks one target or history: it must be non-empty, hold
// wantLength steps, contain only values float32 can represent (or NaN), and
// hold at least one observation.
func checkHistory(history []float64, wantLength int) error {
	if len(history) == 0 {
		return ErrEmptyHistory
	}
	if len(history) != wantLength {
		return fmt.Errorf("%w: %d steps, want %d", ErrMismatchedLengths, len(history), wantLength)
	}
	if err := checkRepresentable(history); err != nil {
		return err
	}
	for _, value := range history {
		if !math.IsNaN(value) {
			return nil
		}
	}
	return ErrNoObservations
}

// checkRepresentable returns an error wrapping ErrInvalidValue at the first
// value in values that is infinite or whose magnitude exceeds
// math.MaxFloat32. The model computes in float32, where such a value becomes
// an infinity and turns its whole series' forecast into NaN. NaN is allowed.
func checkRepresentable(values []float64) error {
	for i, value := range values {
		if math.Abs(value) > math.MaxFloat32 {
			return fmt.Errorf("%w: %v at step %d", ErrInvalidValue, value, i)
		}
	}
	return nil
}

// validateCovariates checks a task's covariates: names follow the structural
// rules, past series hold pastLength steps, future series hold horizon
// steps, and numeric values are representable in float32.
func validateCovariates(covariates Covariates, pastLength, horizon int) error {
	if err := checkCovariateNames(covariates); err != nil {
		return err
	}
	for _, name := range sortedKeys(covariates.Past) {
		series := covariates.Past[name]
		if err := checkCovariateSeries(name, "past", len(series), pastLength, series); err != nil {
			return err
		}
	}
	for _, name := range sortedKeys(covariates.Future) {
		series := covariates.Future[name]
		if err := checkCovariateSeries(name, "future", len(series), horizon, series); err != nil {
			return err
		}
	}
	for _, name := range sortedKeys(covariates.PastCategorical) {
		series := covariates.PastCategorical[name]
		if err := checkCovariateSeries(name, "past", len(series), pastLength, nil); err != nil {
			return err
		}
	}
	for _, name := range sortedKeys(covariates.FutureCategorical) {
		series := covariates.FutureCategorical[name]
		if err := checkCovariateSeries(name, "future", len(series), horizon, nil); err != nil {
			return err
		}
	}
	return nil
}

// checkCovariateSeries checks one covariate series of length steps against
// wantLength and, for numeric series, rejects values float32 cannot
// represent. numeric is nil for categorical series.
func checkCovariateSeries(name, period string, length, wantLength int, numeric []float64) error {
	if length != wantLength {
		return fmt.Errorf("covariate %q: %w: %s series has %d steps, want %d", name, ErrMismatchedLengths, period, length, wantLength)
	}
	if err := checkRepresentable(numeric); err != nil {
		return fmt.Errorf("covariate %q: %s series: %w", name, period, err)
	}
	return nil
}

// checkCovariateNames returns an error wrapping ErrInvalidCovariates when a
// future key lacks a past key or a name appears in both the numeric and the
// categorical maps.
func checkCovariateNames(covariates Covariates) error {
	for _, name := range sortedKeys(covariates.Future) {
		if _, ok := covariates.Past[name]; !ok {
			return fmt.Errorf("%w: future covariate %q has no past values", ErrInvalidCovariates, name)
		}
	}
	for _, name := range sortedKeys(covariates.FutureCategorical) {
		if _, ok := covariates.PastCategorical[name]; !ok {
			return fmt.Errorf("%w: future categorical covariate %q has no past values", ErrInvalidCovariates, name)
		}
	}
	for _, name := range sortedKeys(covariates.Past) {
		if _, ok := covariates.PastCategorical[name]; ok {
			return fmt.Errorf("%w: covariate %q is both numeric and categorical", ErrInvalidCovariates, name)
		}
	}
	return nil
}

// checkCrossLearningFits returns an error wrapping
// ErrCrossLearningBatchTooLarge when the tasks' rows together exceed
// batchSize, since cross-learning must place the whole call in one pass.
func checkCrossLearningFits(tasks []Task, batchSize int) error {
	rows := 0
	for _, task := range tasks {
		rows += task.rowCount()
	}
	if rows > batchSize {
		return fmt.Errorf("%w: %d rows, batch size %d", ErrCrossLearningBatchTooLarge, rows, batchSize)
	}
	return nil
}

// sortedKeys returns m's keys in ascending order, so validation reports the
// same violation on every run when a task has several.
func sortedKeys[V any](m map[string]V) []string {
	return slices.Sorted(maps.Keys(m))
}
