# chronos2-go

Probabilistic time series forecasting in pure Go, powered by Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) foundation model running entirely on-device.

> [!NOTE]
> `chronos2-go` is at v1, and its API is stable: later v1 releases add capabilities and fixes without breaking code written against earlier ones.

Hand it a history and a horizon, and it returns a forecast: a median line plus percentile bands describing the range of likely outcomes. There is nothing to train, no service to call, and no Python runtime; the model's weights are embedded in your binary.

```go
m, err := small.New()
if err != nil {
    return err
}

fc, err := m.Forecast(ctx, history, 48)
if err != nil {
    return err
}

median := fc.Median()        // the most likely path, 48 steps
low := fc.Percentile10()     // 80% of outcomes should fall
high := fc.Percentile90()    // between these two lines
```


## Features

- **Zero-Shot Forecasting:** Chronos-2 is pretrained on a broad mix of real and synthetic series, so it forecasts new data without fitting or tuning.
- **Self-Contained:** pure Go with no cgo, no network access, and no external runtime. Weights ship inside your binary.
- **Probabilistic Output:** every forecast carries a median and percentile bands from the 1st to the 99th percentile.
- **Simple by Default:** forecasting one series is one method call; nothing more is required.
- **More When You Need It:** forecast many series at once, forecast related series jointly, inform a forecast with covariates such as weather or promotions, and extract embeddings for clustering or similarity search.
- **Missing Data Handled:** mark gaps with `NaN`; the model works around them.
- **Two Model Sizes:** choose by import path, and pay in binary size only for the model you use.
- **Safe for Concurrency:** one model value serves any number of goroutines.


## Models

| Module | Package | Checkpoint | Parameters | Adds to Binary |
|---|---|---|---|---|
| [`github.com/pinkhop/chronos2-go-small`](https://github.com/pinkhop/chronos2-go-small) | `small` | `autogluon/chronos-2-small` | ~28M | ~112 MB |
| [`github.com/pinkhop/chronos2-go-base`](https://github.com/pinkhop/chronos2-go-base) | `base` | `amazon/chronos-2` | ~120M | ~478 MB |

Both models read up to 8,192 steps of history and forecast up to 1,024 steps in a single pass. The base model is more accurate and predicts more percentile levels natively; the small model is faster and lighter. Switching between them means changing one import, with no other code changes.

Each model lives in a repository of its own, because the two together exceed the size limit Go places on a repository's download. A model's package name (`small` or `base`) differs from the last element of its import path, so import it with that name as an alias, as the examples below do.


## Installation

Add the model you want; it brings the core library with it:

```console
$ go get github.com/pinkhop/chronos2-go-small
```

Each model module is versioned independently of the library and requires the oldest library release it works with, so library fixes never wait on a model release. To take the newest library release, update it directly:

```console
$ go get github.com/pinkhop/chronos2-go@latest
```

Requires Go 1.27.4 or later.


## Usage

### Forecast a Series

```go
import (
    small "github.com/pinkhop/chronos2-go-small"
)

m, err := small.New()
if err != nil {
    return fmt.Errorf("load model: %w", err)
}

fc, err := m.Forecast(ctx, hourlyLoad, 24)
if err != nil {
    return fmt.Errorf("forecast load: %w", err)
}
```

`small.New()` returns the same shared model on every call, so call it wherever convenient.

### Read the Results

A forecast holds one line per percentile, each as long as the horizon:

```go
median := fc.Median()            // the point forecast
p90 := fc.Percentile90()         // a natively predicted percentile
p925, err := fc.Percentile(92.5) // any percentile from 1 to 99, interpolated as needed
```

Named methods exist for each percentile the base model predicts (`Percentile1`, `Percentile5`, `Percentile10`, …, `Percentile95`, `Percentile99`). Each returns `nil` when the model in use does not predict that level directly; `Percentile(p)` always works within 1–99.

### Missing Values and Long Histories

Mark gaps with `math.NaN()`. Histories longer than the model's window are trimmed to the most recent 8,192 steps automatically; pass `chronos2.WithContextLength(n)` to use a shorter window for speed.

### Beyond One Series

```go
// Many independent series at once.
fcs, err := m.ForecastBatch(ctx, [][]float64{storeA, storeB, storeC}, 48)

// Related series forecast jointly, such as CPU, memory, and disk for one host.
fcs, err := m.ForecastMultivariate(ctx, [][]float64{cpu, mem, disk}, 48)

// A target informed by covariates: past-only, or known into the future.
fc, err := m.ForecastWithCovariates(ctx, chronos2.CovariateTask{
    Target: sales,
    Past:   map[string][]float64{"temperature": pastTemps},
    Future: map[string][]float64{"temperature": forecastTemps},
}, 48)

// A vector representation of a series, for clustering or similarity search.
emb, err := m.Embed(ctx, history)
vector := emb.Summary()
```

`ForecastTasks` combines all of these in one call, including categorical covariates and cross-learning across series.

### Options

| Option | Effect |
|---|---|
| `chronos2.WithContextLength(n)` | Use only the most recent `n` steps of history. |
| `chronos2.WithMaxParallelism(n)` | Cap the goroutines one call uses (default: all CPUs). |
| `chronos2.WithLongHorizon()` | Allow horizons beyond 1,024 steps, using a slower, less accurate heuristic. |
| `chronos2.WithCrossLearning()` | Let a batch of series inform one another's forecasts. |
| `chronos2.WithBatchSize(n)` | Cap how many series are processed per pass, bounding memory. |


## Command-Line Tool

The `chronos2` command is a quick way to try the library without writing any Go: it forecasts a CSV column from the terminal, with both models built in. It is not meant as a production forecasting tool; programs should call the library directly.

```console
$ go install github.com/pinkhop/chronos2-go/cmd/chronos2@latest
$ chronos2 forecast --model small --horizon 24 --percentiles 10,90 < load.csv
step,p10,median,p90
1,410.2,455.8,503.1
2,398.7,449.0,498.4
```

Because it embeds both models, the binary is about 600 MB.


## Performance

The library ships in two builds. The default build's numeric kernels are portable Go and run on every platform Go supports. The SIMD build, available on arm64 and amd64, replaces the hottest kernels (matrix multiplication, softmax, and the element-wise bias and ReLU passes) with versions that use the CPU's vector (SIMD) instructions.

On an Apple M2 Max, a univariate forecast from an 8,192-step history, 64 steps ahead, and an embedding of a 2,048-step history take:

| Model | Operation | Default, One Goroutine | SIMD, One Goroutine | Default, All 12 Cores | SIMD, All 12 Cores |
|---|---|---|---|---|---|
| small | Forecast | 1.18 s | 401 ms (2.9×) | 162 ms | 65 ms (2.5×) |
| small | Embedding | 270 ms | 94 ms (2.9×) | 42 ms | 21 ms (2.1×) |
| base | Forecast | 4.84 s | 1.63 s (3.0×) | 647 ms | 254 ms (2.6×) |
| base | Embedding | 1.18 s | 400 ms (2.9×) | 163 ms | 70 ms (2.3×) |

On an Intel Xeon Platinum 8358 virtual machine with 8 vCPUs, where the SIMD build uses AVX-512, the same operations take:

| Model | Operation | Default, One Goroutine | SIMD, One Goroutine | Default, All 8 vCPUs | SIMD, All 8 vCPUs |
|---|---|---|---|---|---|
| small | Forecast | 6.92 s | 657 ms (10.5×) | 1.81 s | 182 ms (10.0×) |
| small | Embedding | 1.56 s | 160 ms (9.7×) | 407 ms | 56 ms (7.3×) |
| base | Forecast | 28.9 s | 2.68 s (10.8×) | 7.28 s | 670 ms (10.9×) |
| base | Embedding | 6.69 s | 676 ms (9.9×) | 1.73 s | 172 ms (10.1×) |

The library uses every core by default; `chronos2.WithMaxParallelism` limits it. Each call allocates 15–40 MB of working memory on top of the embedded weights, which `Load` uses in place rather than copying. Shorter histories are proportionally faster: `chronos2.WithContextLength` trades some accuracy for speed.

### The SIMD Build

In the benchmarks above, the SIMD build makes forecasts and embeddings 2.9–3.0× faster on one goroutine and 2.1–2.6× faster across all cores on the M2 Max, and 9.7–10.8× faster on one goroutine and 7.3–10.9× faster across all vCPUs on the Xeon. It runs on any 64-bit Arm processor, including Apple silicon and Arm servers such as AWS Graviton, because every such processor has the NEON vector instructions it uses. On amd64, it checks the CPU once at startup and uses AVX-512 where available, AVX2 with FMA on other processors from Intel Haswell and AMD Zen onward, and the portable kernels on older processors.

Enable it by setting the `GOEXPERIMENT=simd` environment variable whenever the Go toolchain compiles your program:

```console
$ GOEXPERIMENT=simd go build ./...
$ GOEXPERIMENT=simd go test ./...
```

To make it the default for every build on your machine, run `go env -w GOEXPERIMENT=simd`. The setting is recorded in the binary, so `go version -m ./yourprogram` shows `build GOEXPERIMENT=simd` when the SIMD build is in use. Setting the build tag with `-tags goexperiment.simd` instead does not work: the program fails to link.

Nothing in your code changes. Forecasts from the SIMD build match the default build's within float32 rounding, not bit for bit, and so do forecasts from amd64 processors with and without AVX-512. On architectures other than arm64 and amd64, the setting has no effect on this library, which builds with its portable kernels.

`GOEXPERIMENT` is a toolchain-wide setting: it applies to every package in your binary, and it enables Go's experimental `simd/archsimd` package, which is outside the Go 1 compatibility promise and may change in any Go release. A Go upgrade can therefore break the SIMD build while the default build keeps working.


## How It Works

Chronos-2 is a transformer encoder that reads a series in 16-step patches and predicts every future patch's percentiles in one pass. Its group attention lets related series — multivariate targets and covariates — exchange information. This library reimplements the model's forward pass in Go, reads the published weights directly, and is verified against the reference PyTorch implementation's outputs for every supported capability on both models. The one deliberate difference concerns constant series, such as a static categorical covariate: the library normalizes them exactly, where PyTorch's float32 rounding can turn them into noise. [`docs/DESIGN.md`](docs/DESIGN.md) covers the design in depth.


## Contributing

[`AGENTS.md`](AGENTS.md) describes the repository layout, the `make` targets, and the rules every change follows, for human and AI contributors alike. The model weights live in [`chronos2-go-base`](https://github.com/pinkhop/chronos2-go-base) and [`chronos2-go-small`](https://github.com/pinkhop/chronos2-go-small), each with an `AGENTS.md` of its own.


## License

Licensed under the Apache License, Version 2.0; see [`LICENSE`](LICENSE).

Chronos-2 was created by Amazon, and its weights are distributed under Apache-2.0; see [`NOTICE`](NOTICE). If you use Chronos-2 in research, cite the authors' paper, [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821).
