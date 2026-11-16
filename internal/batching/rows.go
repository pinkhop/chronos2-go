package batching

import "slices"

// row is one engine row before trimming and padding.
type row struct {
	// past holds the row's full supplied history in float32.
	past []float32

	// future holds the row's Horizon future values, or nil when the row has no
	// known future (targets and past-only covariates).
	future []float32
}

// preparedTask is a task converted to engine rows in reference order.
type preparedTask struct {
	// rows holds the targets, then the past-only covariates, then the
	// known-future covariates, with covariates ordered by name in each class.
	rows []row

	// targets is the number of leading rows that are targets.
	targets int
}

// prepareTask converts task into engine rows: targets first, then past-only
// covariates, then known-future covariates, each class sorted by name with
// numeric and categorical covariates interleaved, as the reference orders a
// single covariate dictionary. categories supplies the call-wide ordinal codes.
func prepareTask(task Task, categories ordinalCategories) preparedTask {
	pastOnly, knownFuture := covariateNames(task)
	rows := make([]row, 0, len(task.Targets)+len(pastOnly)+len(knownFuture))
	for _, target := range task.Targets {
		rows = append(rows, row{past: toFloat32(target)})
	}
	for _, name := range slices.Concat(pastOnly, knownFuture) {
		rows = append(rows, covariateRow(task, name, categories))
	}
	return preparedTask{rows: rows, targets: len(task.Targets)}
}

// taskRowCounts returns the number of engine rows prepareTask produces for
// each task, without preparing any: one per target and one per covariate.
func taskRowCounts(tasks []Task) []int {
	counts := make([]int, len(tasks))
	for i, task := range tasks {
		counts[i] = len(task.Targets) + len(task.PastNumeric) + len(task.PastCategorical)
	}
	return counts
}

// covariateNames returns task's past-only and known-future covariate names,
// each sorted. A covariate is known-future when its name has future values in
// the numeric or categorical future map.
func covariateNames(task Task) (pastOnly, knownFuture []string) {
	classify := func(name string, hasFuture bool) {
		if hasFuture {
			knownFuture = append(knownFuture, name)
		} else {
			pastOnly = append(pastOnly, name)
		}
	}
	for name := range task.PastNumeric {
		_, hasFuture := task.FutureNumeric[name]
		classify(name, hasFuture)
	}
	for name := range task.PastCategorical {
		_, hasFuture := task.FutureCategorical[name]
		classify(name, hasFuture)
	}
	slices.Sort(pastOnly)
	slices.Sort(knownFuture)
	return pastOnly, knownFuture
}

// covariateRow returns the engine row for task's covariate name, encoding it
// first when it is categorical.
func covariateRow(task Task, name string, categories ordinalCategories) row {
	if values, ok := task.PastNumeric[name]; ok {
		return row{past: toFloat32(values), future: toFloat32(task.FutureNumeric[name])}
	}
	past, future := encodeCategorical(task, name, categories)
	return row{past: past, future: future}
}

// toFloat32 converts values to float32, keeping NaN. It returns nil for an
// empty or nil slice.
func toFloat32(values []float64) []float32 {
	if len(values) == 0 {
		return nil
	}
	converted := make([]float32, len(values))
	for i, value := range values {
		converted[i] = float32(value)
	}
	return converted
}
