package parallel

import (
	"slices"
	"sync"
	"testing"
)

// indexRange is one [start, end) range passed to a body.
type indexRange struct {
	// start and end bound the range: start inclusive, end exclusive.
	start, end int
}

// rangeRecorder is a spy body that records every range it receives; it is
// safe for concurrent use.
type rangeRecorder struct {
	// mutex guards ranges against concurrent bodies.
	mutex sync.Mutex

	// ranges holds every received range in arrival order.
	ranges []indexRange
}

// body records one call; pass it to For.
func (r *rangeRecorder) body(start, end int) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.ranges = append(r.ranges, indexRange{start, end})
}

// sorted returns the recorded ranges ordered by start; call it only after For
// returns.
func (r *rangeRecorder) sorted() []indexRange {
	return slices.SortedFunc(slices.Values(r.ranges), func(a, b indexRange) int { return a.start - b.start })
}

// assertTiles fails the test unless ranges are non-empty, contiguous, and
// exactly cover [0, n).
func assertTiles(t *testing.T, ranges []indexRange, n int) {
	t.Helper()
	next := 0
	for _, r := range ranges {
		if r.start != next || r.end <= r.start {
			t.Fatalf("ranges %v do not tile [0, %d)", ranges, n)
		}
		next = r.end
	}
	if next != n {
		t.Fatalf("ranges %v end at %d, want %d", ranges, next, n)
	}
}

// TestFor_SeveralWorkers_TilesTheRangeInWorkersParts verifies that For, given several workers, tiles the range in workers parts.
func TestFor_SeveralWorkers_TilesTheRangeInWorkersParts(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(10, 3, recorder.body)

	// Then
	ranges := recorder.sorted()
	assertTiles(t, ranges, 10)
	if len(ranges) != 3 {
		t.Errorf("got %d ranges, want 3", len(ranges))
	}
}

// TestFor_UnevenSplit_RangeSizesDifferByAtMostOne verifies that For, given uneven split, range sizes differ by at most one.
func TestFor_UnevenSplit_RangeSizesDifferByAtMostOne(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(11, 4, recorder.body)

	// Then
	for _, r := range recorder.sorted() {
		if size := r.end - r.start; size < 2 || size > 3 {
			t.Errorf("range %v has size %d, want 2 or 3", r, size)
		}
	}
}

// TestFor_MoreWorkersThanItems_UsesOneRangePerItem verifies that For, given more workers than items, uses one range per item.
func TestFor_MoreWorkersThanItems_UsesOneRangePerItem(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(3, 8, recorder.body)

	// Then
	ranges := recorder.sorted()
	assertTiles(t, ranges, 3)
	if len(ranges) != 3 {
		t.Errorf("got %d ranges, want 3", len(ranges))
	}
}

// TestFor_OneWorker_CallsBodyOnceWithWholeRange verifies that For, given one worker, calls body once with whole range.
func TestFor_OneWorker_CallsBodyOnceWithWholeRange(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(7, 1, recorder.body)

	// Then
	if want := []indexRange{{0, 7}}; !slices.Equal(recorder.ranges, want) {
		t.Errorf("ranges = %v, want %v", recorder.ranges, want)
	}
}

// TestFor_NonPositiveWorkers_CallsBodyOnceWithWholeRange verifies that For, given non-positive workers, calls body once with the whole range.
func TestFor_NonPositiveWorkers_CallsBodyOnceWithWholeRange(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(5, 0, recorder.body)

	// Then
	if want := []indexRange{{0, 5}}; !slices.Equal(recorder.ranges, want) {
		t.Errorf("ranges = %v, want %v", recorder.ranges, want)
	}
}

// TestFor_EmptyRange_NeverCallsBody verifies that For, given empty range, never calls body.
func TestFor_EmptyRange_NeverCallsBody(t *testing.T) {
	t.Parallel()

	// Given
	var recorder rangeRecorder

	// When
	For(0, 4, recorder.body)

	// Then
	if len(recorder.ranges) != 0 {
		t.Errorf("ranges = %v, want none", recorder.ranges)
	}
}

// TestFor_BodyWritesDisjointIndices_AllWritesVisibleOnReturn verifies that For, given body writes disjoint indices, all writes visible on return.
func TestFor_BodyWritesDisjointIndices_AllWritesVisibleOnReturn(t *testing.T) {
	t.Parallel()

	// Given
	const n = 1000
	values := make([]int, n)

	// When
	For(n, 7, func(start, end int) {
		for i := start; i < end; i++ {
			values[i] = i * i
		}
	})

	// Then
	for i, value := range values {
		if value != i*i {
			t.Fatalf("values[%d] = %d, want %d", i, value, i*i)
		}
	}
}

// TestFor_BodyPanicsOnWorker_PanicReachesCaller verifies that For, given body panics on worker, panic reaches caller.
func TestFor_BodyPanicsOnWorker_PanicReachesCaller(t *testing.T) {
	t.Parallel()

	// Given
	const failure = "range failed"
	var recovered any

	// When
	func() {
		defer func() { recovered = recover() }()
		For(8, 4, func(start, _ int) {
			if start == 0 {
				panic(failure)
			}
		})
	}()

	// Then
	if recovered != failure {
		t.Errorf("recovered %v, want %q", recovered, failure)
	}
}
