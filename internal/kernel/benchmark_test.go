package kernel

import (
	"fmt"
	"testing"
)

// Real Chronos-2 dimensions: the base and small model and feed-forward
// widths, the 64-wide attention heads, and 1024 tokens as a representative
// number of rows flowing through each projection.
const (
	benchmarkTokens       = 1024
	baseModelWidth        = 768
	baseFeedForwardWidth  = 3072
	smallModelWidth       = 512
	smallFeedForwardWidth = 2048
	headWidth             = 64
	baseHeads             = 12
	// attentionSequence is a full 8192-step context: 512 patches, the [REG]
	// token, and 64 future patches.
	attentionSequence = 577
	// attentionQueryBlock mirrors queryBlock in internal/network: time
	// attention scores at most this many queries per MatMulTransposed call.
	attentionQueryBlock = 64
)

// reportFlops records the benchmark's arithmetic rate, counting a multiply
// and an add per inner-product term.
func reportFlops(b *testing.B, m, k, n int) {
	b.Helper()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

// BenchmarkMatMulTransposed measures the MatMulTransposed kernel at the
// projection shapes of both models and at the two shapes time attention uses
// per query block: scoring a block of queries against every key, then
// applying those scores to the transposed values.
func BenchmarkMatMulTransposed(b *testing.B) {
	shapes := []struct{ m, k, n int }{
		{benchmarkTokens, baseModelWidth, baseModelWidth},
		{benchmarkTokens, baseModelWidth, baseFeedForwardWidth},
		{benchmarkTokens, baseFeedForwardWidth, baseModelWidth},
		{benchmarkTokens, smallModelWidth, smallModelWidth},
		{benchmarkTokens, smallModelWidth, smallFeedForwardWidth},
		{attentionSequence, headWidth, attentionSequence},
		{attentionQueryBlock, headWidth, attentionSequence},
		{attentionQueryBlock, attentionSequence, headWidth},
	}
	for _, shape := range shapes {
		b.Run(fmt.Sprintf("m=%d,k=%d,n=%d", shape.m, shape.k, shape.n), func(b *testing.B) {
			left := randomValues(1, shape.m*shape.k)
			right := randomValues(2, shape.n*shape.k)
			out := make([]float32, shape.m*shape.n)
			for b.Loop() {
				MatMulTransposed(out, left, right, shape.m, shape.k, shape.n)
			}
			reportFlops(b, shape.m, shape.k, shape.n)
		})
	}
}

// BenchmarkMatMul measures the MatMul kernel.
func BenchmarkMatMul(b *testing.B) {
	const m, k, n = attentionSequence, attentionSequence, headWidth
	left := randomValues(3, m*k)
	right := randomValues(4, k*n)
	out := make([]float32, m*n)
	for b.Loop() {
		MatMul(out, left, right, m, k, n)
	}
	reportFlops(b, m, k, n)
}

// BenchmarkRMSNorm measures the RMSNorm kernel.
func BenchmarkRMSNorm(b *testing.B) {
	x := randomValues(5, benchmarkTokens*baseModelWidth)
	weight := randomValues(6, baseModelWidth)
	out := make([]float32, len(x))
	for b.Loop() {
		RMSNorm(out, x, weight, benchmarkTokens, baseModelWidth, 1e-6)
	}
}

// BenchmarkSoftmaxRows measures the SoftmaxRows kernel.
func BenchmarkSoftmaxRows(b *testing.B) {
	x := randomValues(7, attentionSequence*attentionSequence)
	for b.Loop() {
		SoftmaxRows(x, attentionSequence, attentionSequence)
	}
}

// BenchmarkApplyRoPE measures the ApplyRoPE kernel.
func BenchmarkApplyRoPE(b *testing.B) {
	x := randomValues(8, attentionSequence*baseHeads*headWidth)
	cos, sin := ropeTables(attentionSequence, headWidth, 10000)
	for b.Loop() {
		ApplyRoPE(x, cos, sin, attentionSequence, baseHeads, headWidth)
	}
}

// BenchmarkAddBias measures the AddBias kernel.
func BenchmarkAddBias(b *testing.B) {
	x := randomValues(9, benchmarkTokens*baseModelWidth)
	bias := randomValues(10, baseModelWidth)
	for b.Loop() {
		AddBias(x, bias, benchmarkTokens, baseModelWidth)
	}
}

// BenchmarkAddBiasReLU measures the AddBiasReLU kernel at the base model's
// feed-forward width.
func BenchmarkAddBiasReLU(b *testing.B) {
	x := randomValues(16, benchmarkTokens*baseFeedForwardWidth)
	bias := randomValues(17, baseFeedForwardWidth)
	for b.Loop() {
		AddBiasReLU(x, bias, benchmarkTokens, baseFeedForwardWidth)
	}
}

// BenchmarkReLU measures the ReLU kernel.
func BenchmarkReLU(b *testing.B) {
	x := randomValues(11, benchmarkTokens*baseFeedForwardWidth)
	for b.Loop() {
		ReLU(x)
	}
}

// BenchmarkAdd measures the Add kernel.
func BenchmarkAdd(b *testing.B) {
	x := randomValues(12, benchmarkTokens*baseModelWidth)
	y := randomValues(13, benchmarkTokens*baseModelWidth)
	for b.Loop() {
		Add(x, x, y)
	}
}

// BenchmarkDot measures the Dot kernel.
func BenchmarkDot(b *testing.B) {
	x := randomValues(14, baseFeedForwardWidth)
	y := randomValues(15, baseFeedForwardWidth)
	var sink float32
	for b.Loop() {
		sink += Dot(x, y)
	}
	_ = sink
}
