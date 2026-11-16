package parity

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/pinkhop/chronos2-go/internal/network"
)

// benchmarkOutputPatches is the number of output patches per benchmark pass,
// a 64-step horizon.
const benchmarkOutputPatches = 4

// benchmarkBatch returns a batch of rows sine-wave series of length steps,
// each in its own group, with no known future values.
func benchmarkBatch(rows, length int) network.Batch {
	context := make([]float32, rows*length)
	for i := range context {
		context[i] = float32(math.Sin(float64(i%length)/7) + float64(i/length))
	}
	future := make([]float32, rows*benchmarkOutputPatches*16)
	for i := range future {
		future[i] = float32(math.NaN())
	}
	groups := make([]int, rows)
	for i := range groups {
		groups[i] = i
	}
	return network.Batch{
		Rows: rows, ContextLength: length, Context: context, GroupIDs: groups,
		OutputPatches: benchmarkOutputPatches, FutureCovariates: future,
	}
}

// forwardBenchmarkCase is one benchmarked batch shape and parallelism.
type forwardBenchmarkCase struct {
	// rows is the batch's row count.
	rows int
	// length is each row's context length in steps.
	length int
	// parallelism is ForwardOptions.Parallelism.
	parallelism int
}

// forwardBenchmarkCases covers short and full contexts, one and sixteen
// rows, and serial and fully parallel passes. Sixteen rows run only in
// parallel, since a serial pass of that size takes tens of seconds.
func forwardBenchmarkCases() []forwardBenchmarkCase {
	all := runtime.GOMAXPROCS(0)
	return []forwardBenchmarkCase{
		{1, 2048, 1},
		{1, 2048, all},
		{1, 8192, 1},
		{1, 8192, all},
		{16, 2048, all},
		{16, 8192, all},
	}
}

// benchmarkForward runs every forward benchmark case against the named model.
func benchmarkForward(b *testing.B, model string) {
	net := realNetwork(b, model)
	for _, c := range forwardBenchmarkCases() {
		b.Run(fmt.Sprintf("rows=%d,context=%d,parallelism=%d", c.rows, c.length, c.parallelism), func(b *testing.B) {
			batch := benchmarkBatch(c.rows, c.length)
			options := network.ForwardOptions{Parallelism: c.parallelism}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := net.Forward(b.Context(), batch, options); err != nil {
					b.Fatalf("Forward: %v", err)
				}
			}
		})
	}
}

// BenchmarkNetworkForward_Small measures one small-model forward pass.
func BenchmarkNetworkForward_Small(b *testing.B) {
	benchmarkForward(b, "small")
}

// BenchmarkNetworkForward_Base measures one base-model forward pass.
func BenchmarkNetworkForward_Base(b *testing.B) {
	benchmarkForward(b, "base")
}
