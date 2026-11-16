package parity

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"testing"

	small "github.com/pinkhop/chronos2-go-small"
)

// heapProbeEnvironment names the environment variable that turns
// TestSmallNewHeapProbe from a no-op into the measuring child process.
const heapProbeEnvironment = "CHRONOS2_PARITY_HEAP_PROBE"

// heapProbeTest is the test the child process runs.
const heapProbeTest = "^TestSmallNewHeapProbe$"

// maxAliasedLoadHeapBytes bounds the heap small.New may allocate when it
// views the embedded weights in place. Copying them allocates the full
// checkpoint, about 112 MB, so 16 MB separates the two cases with ample
// room for the network's own bookkeeping.
const maxAliasedLoadHeapBytes = 16 << 20

// TestSmallNew_EmbeddedWeights_AreViewedNotCopied verifies that loading the
// embedded small model on a little-endian platform views its weights in
// place: the model packages must embed shards in memory the linker aligns,
// or Load silently falls back to copying every weight. small.New caches its
// model per process, so the measurement runs in a fresh child process.
func TestSmallNew_EmbeddedWeights_AreViewedNotCopied(t *testing.T) {
	t.Parallel()

	// Given
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		t.Skip("weights are viewed in place only on little-endian platforms")
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test binary: %v", err)
	}
	// #nosec G204 -- re-runs this test binary with a fixed test filter.
	child := exec.CommandContext(t.Context(), executable, "-test.run="+heapProbeTest, "-test.count=1")
	child.Env = append(os.Environ(), heapProbeEnvironment+"=1")

	// When
	output, err := child.Output()
	// Then
	if err != nil {
		t.Fatalf("heap probe failed: %v\n%s", err, output)
	}
	growth, err := strconv.ParseUint(string(bytes.TrimSpace(bytes.SplitN(output, []byte("\n"), 2)[0])), 10, 64)
	if err != nil {
		t.Fatalf("parse heap probe output %q: %v", output, err)
	}
	if growth > maxAliasedLoadHeapBytes {
		t.Errorf("small.New grew the heap by %.1f MB, want at most %.1f MB: the weights were copied, not viewed",
			float64(growth)/(1<<20), float64(maxAliasedLoadHeapBytes)/(1<<20))
	}
}

// TestSmallNewHeapProbe is the child half of
// TestSmallNew_EmbeddedWeights_AreViewedNotCopied. It does nothing unless
// heapProbeEnvironment is set; then it loads the small model and prints, as
// the first line of standard output, how many bytes the live heap grew.
func TestSmallNewHeapProbe(t *testing.T) {
	if os.Getenv(heapProbeEnvironment) == "" {
		t.Skip("runs only as the child of TestSmallNew_EmbeddedWeights_AreViewedNotCopied")
	}

	// Given
	before := liveHeapBytes()

	// When
	model, err := small.New()
	// Then
	if err != nil {
		t.Fatalf("small.New() error = %v", err)
	}
	after := liveHeapBytes()
	runtime.KeepAlive(model)
	_, _ = fmt.Fprintln(os.Stdout, max(after, before)-before)
}

// liveHeapBytes returns the bytes of live heap objects after a full garbage
// collection, so garbage from earlier work does not count.
func liveHeapBytes() uint64 {
	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats.HeapAlloc
}
