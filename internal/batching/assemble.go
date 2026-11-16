package batching

import "math"

// taskSpan is a half-open range [start, end) of task indexes packed into one
// batch.
type taskSpan struct {
	// start is the index of the first task in the batch.
	start int

	// end is one past the index of the last task in the batch.
	end int
}

// chunkTasks packs tasks, given each one's row count, in order into spans of
// at most batchSize rows. A task is never split, and a task with more than
// batchSize rows forms a span of its own. Unlike the reference, which keeps
// adding tasks while a batch is below the limit, a span never overshoots
// batchSize by combining tasks.
func chunkTasks(taskRows []int, batchSize int) []taskSpan {
	var spans []taskSpan
	start, rows := 0, 0
	for i, count := range taskRows {
		if i > start && rows+count > batchSize {
			spans = append(spans, taskSpan{start: start, end: i})
			start, rows = i, 0
		}
		rows += count
	}
	if start < len(taskRows) {
		spans = append(spans, taskSpan{start: start, end: len(taskRows)})
	}
	return spans
}

// assembleBatch lays out members, the prepared tasks of one span whose first
// task has index firstTask in the call, as one batch: rows trimmed to the
// context window and left-padded with NaN to the longest, future rows padded
// with NaN to whole output patches, group IDs, and target refs.
func assembleBatch(members []preparedTask, firstTask int, opts Options) Batch {
	rows, contextLength := batchShape(members, opts.ContextLength)
	futureLength := (opts.Horizon + outputPatchSize - 1) / outputPatchSize * outputPatchSize
	batch := Batch{
		Rows:          rows,
		ContextLength: contextLength,
		Context:       nanSlice(rows * contextLength),
		GroupIDs:      make([]int, rows),
		FutureLength:  futureLength,
		Future:        nanSlice(rows * futureLength),
	}

	r := 0
	for offset, task := range members {
		group := offset
		if opts.CrossLearning {
			group = 0
		}
		for i, taskRow := range task.rows {
			if i < task.targets {
				batch.Targets = append(batch.Targets, TargetRef{Task: firstTask + offset, Target: i, Row: r})
			}
			trimmed := trimToWindow(taskRow.past, opts.ContextLength)
			copy(batch.Context[(r+1)*contextLength-len(trimmed):], trimmed)
			copy(batch.Future[r*futureLength:], taskRow.future)
			batch.GroupIDs[r] = group
			r++
		}
	}
	return batch
}

// batchShape returns the row count of members and the length of their longest
// row after trimming to window steps.
func batchShape(members []preparedTask, window int) (rows, contextLength int) {
	for _, task := range members {
		rows += len(task.rows)
		for _, taskRow := range task.rows {
			contextLength = max(contextLength, min(len(taskRow.past), window))
		}
	}
	return rows, contextLength
}

// trimToWindow returns the most recent window steps of history, sharing its
// memory.
func trimToWindow(history []float32, window int) []float32 {
	return history[max(0, len(history)-window):]
}

// nanSlice returns n NaN values.
func nanSlice(n int) []float32 {
	values := make([]float32, n)
	nan := float32(math.NaN())
	for i := range values {
		values[i] = nan
	}
	return values
}
