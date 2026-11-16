package network

import (
	"sync/atomic"

	"github.com/pinkhop/chronos2-go/internal/kernel"
	"github.com/pinkhop/chronos2-go/internal/parallel"
)

// Tile sizes for parallel matrix products.
const (
	// projectionRowTile is the number of tokens in one projection tile. It is
	// a multiple of the generic, NEON, and AVX2 kernels' register-block
	// heights (4, 4, and 2 rows), so their tiles never fall back to the
	// slower remainder path. The AVX-512 kernel's 3-row block leaves one
	// remainder row per tile; tiles of 96, 192, and 384 tokens measured no
	// faster end to end.
	projectionRowTile = 64
	// projectionColumnAlignment is the granularity of a projection tile's
	// width, matching the kernel's column tiling.
	projectionColumnAlignment = 32
	// unitsPerWorker is how many tiles a parallel product aims to create per
	// worker, so dynamic scheduling can even out uneven core speeds.
	unitsPerWorker = 3
)

// forEachUnit calls body once for every unit in [0, units), handing units to
// the pass's workers dynamically: each worker repeatedly claims the next
// unclaimed unit. Faster cores therefore take more units, which keeps
// heterogeneous (performance and efficiency) cores from idling. Results do
// not depend on which worker processes a unit.
func (p *forwardPass) forEachUnit(units int, body func(scratch *workerScratch, unit int)) {
	if units <= 0 {
		return
	}
	var next atomic.Int64
	workers := min(p.workers, units)
	parallel.For(workers, workers, func(int, int) {
		scratch := p.takeScratch()
		defer p.returnScratch(scratch)
		for {
			unit := int(next.Add(1)) - 1
			if unit >= units {
				return
			}
			body(scratch, unit)
		}
	})
}

// takeScratch returns an idle worker scratch, allocating one when none is
// idle.
func (p *forwardPass) takeScratch() *workerScratch {
	select {
	case scratch := <-p.scratch:
		return scratch
	default:
		return p.newScratch()
	}
}

// returnScratch makes scratch available to the next worker. The pool's
// capacity equals the parallelism limit, so the send never blocks.
func (p *forwardPass) returnScratch(scratch *workerScratch) {
	p.scratch <- scratch
}

// projection describes one linear map applied to a block of tokens:
// out = in · weightᵀ, then optionally + bias and ReLU.
type projection struct {
	// out receives rows × outWidth values.
	out []float32
	// in holds rows × inWidth values.
	in []float32
	// weight is [outWidth × inWidth].
	weight []float32
	// bias holds outWidth values, or nil for none.
	bias []float32
	// relu applies ReLU after the bias.
	relu bool
	// rows, inWidth, and outWidth are the product's dimensions.
	rows, inWidth, outWidth int
}

// project computes the projection in parallel tiles of tokens and output
// columns. Splitting columns as well as tokens keeps every worker busy when a
// pass has few tokens, and each weight row is then read by only the workers
// that need it. Every output is computed by one kernel call over the full
// input width, so the result is identical however the work is tiled.
func (p *forwardPass) project(job projection) {
	rowTiles := (job.rows + projectionRowTile - 1) / projectionRowTile
	columnTile := job.outWidth
	if wanted := p.workers * unitsPerWorker; rowTiles < wanted {
		columnTiles := (wanted + rowTiles - 1) / rowTiles
		columnTile = (job.outWidth + columnTiles - 1) / columnTiles
		columnTile = max(projectionColumnAlignment, (columnTile+projectionColumnAlignment-1)/projectionColumnAlignment*projectionColumnAlignment)
		columnTile = min(columnTile, job.outWidth)
	}
	columnTiles := (job.outWidth + columnTile - 1) / columnTile
	p.forEachUnit(rowTiles*columnTiles, func(scratch *workerScratch, unit int) {
		firstRow := unit / columnTiles * projectionRowTile
		firstColumn := unit % columnTiles * columnTile
		projectTile(job, scratch, firstRow, min(projectionRowTile, job.rows-firstRow), firstColumn, min(columnTile, job.outWidth-firstColumn))
	})
}

// projectTile computes rows × columns outputs of job starting at firstRow and
// firstColumn. A full-width tile is written in place; a narrower one is
// computed in scratch and copied into its columns.
func projectTile(job projection, scratch *workerScratch, firstRow, rows, firstColumn, columns int) {
	input := job.in[firstRow*job.inWidth : (firstRow+rows)*job.inWidth]
	weight := job.weight[firstColumn*job.inWidth : (firstColumn+columns)*job.inWidth]
	tile := job.out[firstRow*job.outWidth : (firstRow+rows)*job.outWidth]
	if columns != job.outWidth {
		tile = scratch.tile[:rows*columns]
	}
	kernel.MatMulTransposed(tile, input, weight, rows, job.inWidth, columns)
	// The fused kernel makes one pass over the tile instead of two and gives
	// the same bits as AddBias followed by ReLU.
	switch {
	case job.bias != nil && job.relu:
		kernel.AddBiasReLU(tile, job.bias[firstColumn:firstColumn+columns], rows, columns)
	case job.bias != nil:
		kernel.AddBias(tile, job.bias[firstColumn:firstColumn+columns], rows, columns)
	case job.relu:
		kernel.ReLU(tile)
	}
	if columns == job.outWidth {
		return
	}
	for row := range rows {
		start := (firstRow+row)*job.outWidth + firstColumn
		copy(job.out[start:start+columns], tile[row*columns:(row+1)*columns])
	}
}

// rmsNorm writes the RMS normalization of tokens rows of in into out, in
// parallel; out may alias in.
func (p *forwardPass) rmsNorm(out, in, weight []float32, tokens int) {
	width := p.config.ModelWidth
	parallel.For(tokens, p.workers, func(start, end int) {
		kernel.RMSNorm(out[start*width:end*width], in[start*width:end*width], weight, end-start, width, p.config.LayerNormEpsilon)
	})
}
