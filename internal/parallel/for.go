// Package parallel splits index ranges across a bounded set of goroutines.
// The network uses it to fan a kernel out over rows, heads, or output tiles
// while honoring a call's parallelism limit; the kernels themselves stay
// single-threaded.
package parallel

import "sync"

// For runs body over [0, n) split into at most workers contiguous, non-empty
// ranges whose sizes differ by at most one, and returns once every range has
// finished. Each range runs on its own goroutine except the first, which runs
// on the calling goroutine. When workers <= 1 or n <= 1, For calls body(0, n)
// on the calling goroutine; when n <= 0, it never calls body.
//
// body must be safe to call concurrently on disjoint ranges. Writes made by
// body are visible to the caller when For returns, and no goroutine outlives
// the call. A panic in body propagates to the caller of For after every range
// has stopped; when several ranges panic, one of the panics is raised. body
// must not be nil.
func For(n, workers int, body func(start, end int)) {
	if n <= 0 {
		return
	}
	workers = min(workers, n)
	if workers <= 1 {
		body(0, n)
		return
	}

	// Start every range after the first on its own goroutine.
	var group sync.WaitGroup
	var workerPanic panicRecord
	size, remainder := n/workers, n%workers
	firstEnd := rangeSize(0, size, remainder)
	start := firstEnd
	for worker := 1; worker < workers; worker++ {
		end := start + rangeSize(worker, size, remainder)
		// start is reassigned below, so the goroutine needs its own copy.
		rangeStart := start
		group.Go(func() {
			defer workerPanic.capture()
			body(rangeStart, end)
		})
		start = end
	}

	// Run the first range here, waiting for the workers even if it panics.
	func() {
		defer group.Wait()
		body(0, firstEnd)
	}()
	workerPanic.raise()
}

// rangeSize returns the length of range index when n items are split into
// ranges of size items with the first remainder ranges one item longer.
func rangeSize(index, size, remainder int) int {
	if index < remainder {
		return size + 1
	}
	return size
}

// panicRecord holds the first panic recovered from a worker goroutine so For
// can raise it again on the calling goroutine, where the caller can recover
// it. The zero value is ready to use, and its methods are safe for concurrent
// use.
type panicRecord struct {
	// mutex guards value and panicked.
	mutex sync.Mutex
	// value is the first recovered panic value.
	value any
	// panicked reports whether value holds a recovered panic.
	panicked bool
}

// capture recovers a panic in progress and records it if it is the first.
// It must be called directly by a defer statement, since recover has no
// effect anywhere else.
func (p *panicRecord) capture() {
	recovered := recover()
	if recovered == nil {
		return
	}
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if !p.panicked {
		p.value, p.panicked = recovered, true
	}
}

// raise panics with the recorded value, if any; call it only after every
// worker has finished.
func (p *panicRecord) raise() {
	p.mutex.Lock()
	value, panicked := p.value, p.panicked
	p.mutex.Unlock()
	if panicked {
		panic(value)
	}
}
