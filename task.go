package chronos2

// Covariates holds series that inform a task's targets without being
// forecast, keyed by name. A covariate named only in Past (or
// PastCategorical) is past-only: observed historically and unknown in the
// future, such as yesterday's foot traffic. A covariate named in both Past
// and Future (or both categorical maps) is known-future: the caller supplies
// its values over the horizon, such as a weather forecast or a promotion
// calendar.
//
// Every key in Future must appear in Past, every key in FutureCategorical
// must appear in PastCategorical, and no name may appear in both the numeric
// and the categorical maps; a call that breaks these rules returns
// [ErrInvalidCovariates].
//
// Categorical covariates hold string labels, such as "holiday" or "weekday",
// which are converted to numbers before the model reads them, exactly as
// Chronos-2's own Python pipeline does. In a task with one target, each
// label becomes the average of the target's observed values at the steps
// where the label occurs, smoothed toward the target's overall average; a
// future label never seen in the past becomes that overall average. In a
// task with several targets, each label becomes its position among the
// covariate's distinct labels across the whole call, sorted
// lexicographically; a future label never seen in the past is treated as
// unknown. The empty string is the missing label and counts as a label of
// its own. Conversion uses each task's full history, before trimming to the
// context window.
//
// Any map may be nil. Methods never modify or retain the maps or slices.
type Covariates struct {
	// Past maps numeric covariate names to their histories, each as long as
	// the task's targets. NaN marks a missing value.
	Past map[string][]float64

	// Future maps known-future numeric covariate names to their values over
	// the horizon, each exactly horizon steps long. NaN marks a value unknown
	// at that step.
	Future map[string][]float64

	// PastCategorical maps categorical covariate names to their histories,
	// each as long as the task's targets. The empty string is the missing
	// category, encoded as a category of its own.
	PastCategorical map[string][]string

	// FutureCategorical maps known-future categorical covariate names to
	// their values over the horizon, each exactly horizon steps long.
	FutureCategorical map[string][]string
}

// CovariateTask is one target series informed by covariates, the input of
// [Model.ForecastWithCovariates].
type CovariateTask struct {
	// Target is the series to forecast. NaN marks a missing observation.
	Target []float64

	// Covariates holds the series informing Target.
	Covariates
}

// Task is the general forecasting input of [Model.ForecastTasks]: one or more
// targets the model reads together on one time axis, plus covariates
// informing them. A task's targets inform one another's forecasts, as in
// [Model.ForecastMultivariate].
type Task struct {
	// Targets holds the series to forecast: at least one, all of the same
	// length, each following the history rules of [Model.Forecast].
	Targets [][]float64

	// Covariates holds the series informing Targets.
	Covariates
}

// rowCount returns the number of engine rows the task occupies: its targets
// plus each distinct covariate.
func (t Task) rowCount() int {
	return len(t.Targets) + len(t.Past) + len(t.PastCategorical)
}
