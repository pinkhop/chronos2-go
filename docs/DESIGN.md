# Chronos-2 for Go: Design

This document describes how `github.com/pinkhop/chronos2-go` is built and why: a pure-Go library that runs Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) time series forecasting model on-device, with the model weights embedded in the compiled binary. It covers how the library and its weights are packaged, the internal architecture maintainers implement against, and how the implementation is verified against the reference. The public contract lives in the package documentation (`go doc -all github.com/pinkhop/chronos2-go`), and [`AGENTS.md`](../AGENTS.md) covers the repository's workflow and rules.


## Goals and Non-Goals

### Goals

- Forecast time series with Chronos-2 entirely in-process: no network access, no Python, no cgo, and no external runtime.
- Embed the model weights in the consumer's binary, so a program that imports the library is self-contained.
- Let a project pay only for the model it uses: importing the small model must not pull the base model's weights into the binary or the module download.
- Keep the univariate case — one history in, one forecast out — as simple as a single method call, while still exposing the model's full capability (multivariate targets, covariates, cross-learning, long horizons, and embeddings) through additional methods.
- Reproduce the reference PyTorch implementation's outputs within documented numerical tolerances.
- Isolate the numeric hot paths in small, self-contained kernels whose signatures admit a SIMD implementation selected at build time.

### Non-Goals

- Training or fine-tuning. The library performs inference only.
- GPU or accelerator execution.
- A float32 public API. The internal engine computes in float32, but the API exposes float64 only; see [Public API](#public-api).
- Loading checkpoints (saved copies of trained models) from the network or the file system at run time. Checkpoints arrive as embedded data through model packages.


## Public API

The documentation of package `chronos2` is the public contract: every method, option, result accessor, input rule, and sentinel error is specified there. The API follows these principles, which govern how it may change:

- **Minimal Univariate Path:** `m.Forecast(ctx, history, horizon)` never needs more than that. Every other capability arrives as another method or an option.
- **One Route to the Engine:** `Forecast`, `ForecastBatch`, `ForecastMultivariate`, and `ForecastWithCovariates` convert their input into tasks and follow the same path as `ForecastTasks`.
- **Additive Change Only:** new capabilities arrive as new methods or options; no existing signature changes.
- **Immutable Values:** a `*Model`, a `Forecast`, and an `Embedding` are immutable and safe for concurrent use, and every accessor returns a copy.
- **Classified Errors:** every returned error wraps exactly one sentinel declared in `errors.go`, or the context's error when a call is canceled or times out.
- **Float64 at the Boundary:** the API is float64 because nearly all Go numeric sources produce float64, and converting to the engine's float32 costs microseconds against milliseconds or more of inference. Should a float32 API become necessary, it is reserved as an additive set of `F32`-suffixed methods and types on the same `*Model` (for example, `ForecastF32`), feeding the same internal float32 path. Embeddings stay float32, because they are model-internal representations, not values in the caller's units.


## Packaging

### Modules

The project spans three repositories and five Go modules:

| Module Path | Repository and Directory | Contents |
|---|---|---|
| `github.com/pinkhop/chronos2-go` | `chronos2-go`, root | Package `chronos2`, all internal packages, and a tiny synthetic test checkpoint (saved copy of a model, here randomly initialized rather than trained). No real weights. |
| `github.com/pinkhop/chronos2-go/cmd/chronos2` | `chronos2-go`, `cmd/chronos2` | The `chronos2` command-line tool. |
| `github.com/pinkhop/chronos2-go/parity` | `chronos2-go`, `parity` | Real-model parity tests and their fixtures, the weights-manifest and embed-aliasing checks on the model modules, and the module-size check. Never required by another module and never tagged. |
| `github.com/pinkhop/chronos2-go-base` | `chronos2-go-base`, root | Package `base` and the base checkpoint. |
| `github.com/pinkhop/chronos2-go-small` | `chronos2-go-small`, root | Package `small` and the small checkpoint. |

Each model module is the only module in its repository. That repository holds the module's `New()` source, `weights/`, `LICENSE`, `NOTICE`, `README.md`, `AGENTS.md`, a `Makefile`, and the `mise` files, and nothing else: no tests and no fixtures. Its `make check` target verifies the weights against their manifest and the module against the size limits below. The model packages are named `base` and `small`, which differ from the last element of their import paths, so documentation imports them with an alias, for example `small "github.com/pinkhop/chronos2-go-small"`.

`go get` and `go mod download` fetch whole module archives, so this split also bounds what a consumer downloads: a consumer of the small model never downloads the base weights, and the root module's archive excludes the parity fixtures, which belong to a nested module.

Requirements point in one direction only: each model module requires the root module; the command and `parity` each require the root module and both model modules; and no module requires `parity`. Because Go builds a consumer's dependency graph from `require` directives, `parity` can never enter it. The rule that keeps it that way is that only `parity` imports `parity`.

Each model module requires the root module at the lowest version its `New()` works with, so a consumer's build selects whichever newer library version the consumer or another dependency requires. `parity` replaces the root module with `../`, so it tests the working tree, and takes the model modules from the module cache at the versions it requires, so its real-model tests run against the published weights.

### Size Limits

Go limits a module download in two ways, and the [Go module proxy](https://proxy.golang.org/), which stores mirrors of Go modules as zip archives, enforces the same limits because it runs the same code:

- **Module Archive:** the files in a module's zip archive may total at most 500 MiB uncompressed. The base model module is 477.9 MB, about 44 MiB below the limit; the small model module is 111.8 MB.
- **Repository Archive:** for a module at a repository's root, Go downloads a `git archive` of the whole repository and refuses it when that raw zip exceeds 500 MiB. It applies the limit before it drops the files of nested modules, so they count too. Go disables `export-ignore` attributes for this archive, so no attribute can shrink it.

The two checkpoints together zip to 527.5 MiB (base 427.9 MiB, small 99.6 MiB), more than one repository archive allows, so each lives in a repository of its own, and a model repository holds nothing its module does not need. The repository-archive limit binds this repository too: the root module's download includes `cmd/chronos2`, `parity`, and its fixtures, so no checkpoint may be committed here.

`make test` includes a size check in the `parity` module that covers this repository's published modules, including the root module's whole-repository archive, so an oversized module fails locally or in CI rather than after it is published. It packages each module with `golang.org/x/mod/zip`, the package the `go` command itself uses, so it applies the proxy's exact rules.

### Releases

Versions are independent across repositories. A library release does not require a model release; a model repository releases only when its weights or its `New()` change. Every module's first release is v1.0.0.

A release proceeds in this order, because `go install` checks the command's dependencies against its `go.sum`, which can record a module's checksum only after that module is published:

1.  Tag this repository's commit `vX.Y.Z`, releasing the root module.
2.  For each model whose weights or `New()` changed, run `go mod tidy` in its repository, which records the library's checksum in its `go.sum`; commit, and tag that commit with the model's next version, `vX.Y.Z`.
3.  In `cmd/chronos2`, set the requirements to the released versions, run `go mod tidy`, commit, and tag that commit `cmd/chronos2/vX.Y.Z`.
4.  Set `parity`'s requirements to the same versions and run `go mod tidy` there too, so its real-model tests use the released weights. `parity` is never tagged.

### Embedded Weights

Weights ship as fp32, bit-identical to the published checkpoints. Git LFS is unsuitable because neither the module proxy nor a direct `go get` fetches LFS content: consumers would receive pointer files. Instead, each checkpoint is committed as plain Git files under `weights/` in its model repository:

```text
chronos2-go-small/
├── go.mod · go.sum · small.go
├── LICENSE · NOTICE · README.md · AGENTS.md
├── Makefile · mise.toml · mise.lock
└── weights/
    ├── config.json
    ├── manifest.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

- Each shard is a complete, valid safetensors file holding whole tensors, kept under the shard-size limit (parameter `maxShardBytes`, default 45 MB) so no file reaches GitHub's 50 MB warning threshold.
- `manifest.json` records the upstream repository, revision, and SHA-256 of the original `model.safetensors`, and the SHA-256 of every shard.
- Each shard is embedded as its own `[]byte` variable through `//go:embed`. The Go linker currently aligns embedded `[]byte` data (not a language guarantee; `parity/alias_test.go` verifies it), so `Load` can view the weights in place without copying them. It does not align `string` data, which would leave the weights at whatever byte offset the layout produced and usually force `Load` to copy the whole checkpoint: about 478 MB of heap for the base model. The embedded bytes live in writable memory, so the weights are read-only by convention, not in hardware; nothing may write to a shard after `Load`.
- Each model repository's `.gitattributes` marks `*.safetensors` as binary and keeps it out of any LFS rule.

The upstream checkpoints are:

| Model | Repository | Revision | `model.safetensors` SHA-256 |
|---|---|---|---|
| Base | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` |
| Small | `autogluon/chronos-2-small` | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | `492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd` |

### Loading a Checkpoint

Model packages build their `*Model` through one exported loader in package `chronos2`, `Load`, which takes a `Checkpoint` holding a name, the contents of `config.json`, and the safetensors shards. `Load` parses the configuration, rejecting any patch size or stride other than 16, validates every tensor's name, dtype, and shape against it, and binds the weights. On little-endian platforms, when a tensor's data is 4-byte aligned, the weight slice aliases the embedded bytes directly; otherwise `Load` copies the tensor into an aligned slice. All weights are read-only by convention: the `Model` may retain and alias the shard slices for its whole life, so callers must never modify a shard after passing it to `Load`. Model packages wrap `Load` in `sync.OnceValues`, so decoding happens once per process. `Load` is public so that a separately distributed fine-tuned Chronos-2 checkpoint can ship as its own model package in the same way.

### Licensing

The library is licensed under Apache-2.0, as are both checkpoints and the reference implementation. The root of each repository carries `LICENSE` and a `NOTICE` crediting Amazon's Chronos-2 work. The nested `cmd/chronos2` and `parity` modules carry their own copies of both files, because a nested module's archive contains only the files under its own directory.

### Command-Line Tool

The `chronos2` command is a quick way to try the library from a terminal (forecasting a CSV column without writing any Go), not a production forecasting tool. It covers only univariate forecasting; programs that need forecasts call the library directly. It embeds both models, and it parses its arguments with `github.com/urfave/cli/v3`, the only third-party dependency of any published module. Its `--help` output documents its flags; the rest of its behavior is:

- `forecast` reads one numeric column from `--input` or standard input. A non-numeric first row is treated as a header and skipped. A blank cell or `NaN` marks a missing value; trailing blank lines are ignored.
- It writes CSV to standard output: a `step` column numbered from 1, one `pN` column per requested percentile in ascending order, and a `median` column in its percentile position. Requesting 50 yields only the `median` column.
- `version` prints the module version and VCS revision read from the binary's build information.
- Errors go to standard error with a non-zero exit status.


## Internal Architecture

### How Chronos-2 Works

Chronos-2 is an encoder-only transformer derived from T5. It reads a batch of series and emits quantile forecasts for all of them in one forward pass; there is no autoregressive decoding loop within the model's 1024-step single-pass horizon.

| Hyperparameter | Base | Small |
|---|---|---|
| Model width (`d_model`) | 768 | 512 |
| Feed-forward width (`d_ff`) | 3072 | 2048 |
| Encoder layers | 12 | 6 |
| Attention heads × head width | 12 × 64 | 8 × 64 |
| Quantiles | 21 | 13 |
| Patch size (input and output) | 16 | 16 |
| Max output patches | 64 | 64 |
| Context length | 8192 | 8192 |

One forward pass proceeds as follows. Shapes use _R_ for rows in the batch, _N_ for context patches, _P_ for output patches, _D_ for model width, and _Q_ for quantiles.

1.  **Normalize.** Each row, already trimmed to the context window, is standardized by the mean and population standard deviation (dividing by the count, not the count minus one) of its observed values, then passed through `asinh`. A row with no observed values gets mean 0 and standard deviation 1; a zero standard deviation becomes 1e-5. The mean and standard deviation are kept to invert the transform at the end.
2.  **Patch.** Each row is left-padded with `NaN` to a multiple of 16 and cut into _N_ patches of 16 steps. An observation mask records which values are present; missing values become 0. A patch with no observed values is masked out of attention.
3.  **Embed the context.** Each patch becomes a 48-element vector — 16 time-encoding values (step offsets from −16_N_ to −1, divided by 8192), 16 values, and 16 mask bits — and passes through the input residual block: `W_out·ReLU(W_hidden·x + b_hidden) + b_out + W_res·x + b_res`, yielding a _D_-element token.
4.  **Append the [REG] token.** A learned embedding follows the context tokens and is always attended to.
5.  **Embed the future.** _P_ future patches are built the same way, with time offsets from 0 to 16_P_ − 1. For known-future covariate rows they carry the covariate's future values, transformed exactly as that row's context was (the row's mean and standard deviation, then `asinh`); unknown values, and every value in other rows, become zeros with a zero mask. The same input residual block embeds them. Unlike context patches, future tokens are never masked out of attention, whatever their mask bits say. Each row now holds _S_ = _N_ + 1 + _P_ tokens.
6.  **Encode.** Each of the encoder layers applies three pre-norm residual sublayers:
    - _Time attention_ within each row, across its _S_ tokens, with rotary position embeddings on queries and keys and the row's patch mask applied to keys.
    - _Group attention_ across rows at each token position, restricted to rows in the same group (task) whose token at that position is unmasked. No position embedding applies, so rows form an unordered set.
    - A feed-forward network: `W_out·ReLU(W_in·x)`.
7.  **Final norm.** A final RMS norm over the encoder output.
8.  **Decode.** The last _P_ tokens of each row pass through the output residual block, producing _Q_ × 16 values per patch, quantile-major (the 16 steps of the first quantile, then the 16 steps of the next, so value _q_ × 16 + _step_), reshaped to a _Q_ × 16_P_ quantile grid per row.
9.  **Denormalize.** Each value is mapped back through `sinh`, then multiplied by the row's scale and offset by its mean.

Details the implementation must reproduce exactly:

- Every norm is T5-style RMS normalization: `x · w / sqrt(mean(x²) + 1e-6)`, with no mean subtraction and no bias.
- Attention scores are **not** scaled by 1/√_d_; the model was trained without scaling, and its checkpoint (the saved copy of the trained model) depends on it.
- Masks are additive: masked positions receive the most negative finite float32 value before the softmax, rather than being skipped. In time attention a query always has unmasked keys (the [REG] and future tokens). In group attention, at a context position where a row and every row in its group have no observations, all keys are masked, so the query attends uniformly to every row in the batch, other groups included. Those states never reach a forecast, because masked tokens are excluded as keys everywhere, but they do appear in embeddings (`Patches` and `States`). `Embed` therefore runs each history in a batch of its own, which is also how the reference fixtures are produced.
- Rotary embeddings use θ = 10000 over the 64-wide head with the "rotate half" convention: frequencies are concatenated (not interleaved), and positions count from 0 at the first token of the padded row.
- Linear weights are stored `[out, in]`, as in PyTorch. Attention and feed-forward projections have no bias; the residual blocks' projections do.
- Dropout is inactive at inference and is omitted.
- A constant row (one whose observed values are all equal, as a target-encoded static categorical covariate's row is) normalizes to exactly its value as mean, 1e-5 as standard deviation, and 0 for every normalized input: chronos2-go accumulates the statistics in float64 and rounds once. The reference takes float32 sums, so its mean for such a row can land one unit in the last place off; the 1e-5 floor then does not apply, and its normalized inputs become rounding noise (`asinh(±1)` per value). chronos2-go deliberately does not reproduce this, so a task containing a constant row can differ from the reference by more than the parity tolerance; "Known Deviations" under "Parity Fixtures" lists the affected fixtures.

Group attention's mask for query row _q_ and key row _k_ at token _t_ is `group(q) == group(k) && unmasked(k, t)`. Padding a row to a longer batch shifts its absolute token positions, but rotary embeddings depend only on position differences and padded patches are masked, so a series' forecast does not depend on its batch-mates' lengths beyond floating-point rounding.

### Package Layout

```text
chronos2-go/
├── go.mod · Makefile · mise.toml · mise.lock
├── LICENSE · NOTICE · README.md · AGENTS.md
├── docs/DESIGN.md
├── *.go                    package chronos2: Model, methods, options, errors, results
├── internal/
│   ├── safetensors/        parses safetensors data and exposes typed tensor views
│   ├── kernel/             numeric primitives; the SIMD-replaceable layer
│   ├── parallel/           splits index ranges across a bounded set of goroutines
│   ├── network/            the Chronos-2 forward pass over an engine batch
│   ├── batching/           turns tasks into engine batches: rows, groups, padding, encoding, chunking
│   ├── quantile/           percentile interpolation and weighted quantiles
│   ├── horizon/            long-horizon autoregressive unrolling
│   └── testmodel/          shared test support: the synthetic checkpoint, fixture readers, and parity comparisons
├── testdata/               the synthetic checkpoint and its reference outputs
├── cmd/
│   └── chronos2/           module: the command-line tool
└── parity/                 module: real-model parity tests, the model-module checks, and the module-size check
    └── testdata/
        ├── base/
        └── small/
```

The model packages `base` and `small` live in the `chronos2-go-base` and `chronos2-go-small` repositories, each at its repository's root (see [Embedded Weights](#embedded-weights)).

The root package validates inputs, assembles tasks, invokes the batching, network, and horizon packages, and shapes results. It defines the narrow interfaces it consumes (for example, the network's forward pass), so the batching, horizon, and result logic is testable against fakes without real weights.

### Engine Batch

The boundary between preprocessing and the network is a batch of _R_ rows:

- `Context`: an _R_ × _L_ float32 matrix, row-major, with `NaN` for missing and left padding.
- `GroupIDs`: _R_ integers; rows sharing an ID attend to one another in group attention.
- `FutureCovariates`: an _R_ × 16_P_ float32 matrix, with `NaN` wherever the future value is unknown.
- `OutputPatches`: _P_, from 1 to 64.

The network returns an _R_ × _Q_ × 16_P_ quantile grid in the caller's units and, when embeddings are requested, the _R_ × _S_ × _D_ final hidden states with each row's mean and scale. Rows are self-describing; the network knows nothing about tasks, targets, or covariates.

### Kernels and SIMD Seams

Package `internal/kernel` holds every hot numeric loop. Its functions operate on flat `[]float32` slices with explicit dimensions, write into caller-provided output slices, never allocate, and never start goroutines. That contract lets a SIMD implementation replace any function without touching its callers:

| Function | Operation |
|---|---|
| `MatMulTransposed(out, a, b []float32, m, k, n int)` | `out[m×n] = a[m×k] · b[n×k]ᵀ`, matching PyTorch's `[out, in]` weight layout; also attention scores, `Q · Kᵀ` |
| `MatMul(out, a, b []float32, m, k, n int)` | `out[m×n] = a[m×k] · b[k×n]`, for applying attention weights to values |
| `AddBias(x, bias []float32, rows, cols int)` | add `bias` to every row |
| `RMSNorm(out, x, weight []float32, rows, cols int, eps float32)` | T5-style RMS normalization per row |
| `SoftmaxRows(x []float32, rows, cols int)` | numerically stable softmax per row, in place |
| `ReLU(x []float32)` | in place |
| `AddBiasReLU(x, bias []float32, rows, cols int)` | `AddBias` then `ReLU` in one pass, for projections that feed a ReLU |
| `Add(out, a, b []float32)` | element-wise sum, for residual connections |
| `ApplyRoPE(x, cos, sin []float32, tokens, heads, headDim int)` | rotate-half rotary embedding, in place |
| `Dot(a, b []float32) float32` | inner product |

Each function lives in its own file named for the operation with a `_generic.go` suffix (for example, `matmul_generic.go`), with no build constraint. An operation that has a SIMD implementation spreads across the following files instead, shown here for `MatMulTransposed`:

| File | Build Constraint | Contents |
|---|---|---|
| `matmul_transposed.go` | none | The exported `MatMulTransposed`: its doc comment and length checks, then a call to the unexported `matMulTransposed`. |
| `matmul_transposed_generic.go` | none | `matMulTransposedGeneric` and its helpers, compiled into every build. |
| `matmul_transposed_select_generic.go` | `!goexperiment.simd \|\| !(arm64 \|\| amd64)` | A `matMulTransposed` that calls `matMulTransposedGeneric`. |
| `matmul_transposed_simd_arm64.go` | `goexperiment.simd` | A `matMulTransposed` using NEON instructions from `simd/archsimd`. |
| `matmul_transposed_microkernel_simd_arm64.go` | `goexperiment.simd` | The NEON register-block micro-kernels it calls. |
| `matmul_transposed_simd_amd64.go` | `goexperiment.simd` | A `matMulTransposed` that runs the AVX-512, AVX2, or generic implementation the CPU supports. |
| `matmul_transposed_avx512_simd_amd64.go` | `goexperiment.simd` | `avx512MatMulTransposed`, using AVX-512 instructions. |
| `matmul_transposed_avx512_microkernel_simd_amd64.go` | `goexperiment.simd` | The AVX-512 register-block micro-kernels it calls. |
| `matmul_transposed_avx2_simd_amd64.go` | `goexperiment.simd` | `avx2MatMulTransposed`, using AVX2 and FMA instructions. |
| `matmul_transposed_avx2_microkernel_simd_amd64.go` | `goexperiment.simd` | The AVX2 register-block micro-kernels it calls. |

`SoftmaxRows` follows the same layout (`softmax_rows.go`, `softmax_rows_generic.go`, `softmax_rows_select_generic.go`, `softmax_rows_simd_arm64.go`, and `softmax_rows_simd_amd64.go`), with the vector exponentials it calls in `exp_simd_arm64.go` and `exp_simd_amd64.go` and their shared constants in `exp_constants_simd.go`. `AddBias`, `ReLU`, and `AddBiasReLU` follow it too, each in a single `_simd_<arch>.go` file. On amd64, softmax and the element-wise kernels use AVX2 on CPUs with AVX-512 too, since they are a small share of a forward pass and bound by memory rather than arithmetic. These five are the operations with SIMD implementations; every other kernel is generic in both builds. The vector primitives the amd64 kernels share, such as loads and lane reductions, live in `avx2_simd_amd64.go` and `avx512_simd_amd64.go`, and `vector_level_simd_amd64.go` detects the CPU's vector level.

The `_arm64` and `_amd64` file-name suffixes add the architecture constraint, so the selector's constraint excludes exactly the architectures that have a SIMD file; adding an architecture adds its `_simd_<arch>.go` files and removes that architecture from the selector. A file shared by every SIMD architecture, such as `exp_constants_simd.go`, names those architectures in its build constraint. An implementation whose instructions depend on optional CPU features checks them at run time and calls the generic function when they are absent: amd64 detects its vector level once, at package initialization, and every call in the process uses that level's implementation, so a process never mixes roundings. Keeping the generic code in every build lets the exported doc comment and length checks exist once, and lets the SIMD build test itself against the generic code directly.

Two properties of the Go 1.27 compiler shape the amd64 micro-kernels. It allocates only fifteen vector registers to AVX2 instructions and 31 to AVX-512 instructions, and it implements `MulAdd` with the `VFMADD213PS` form, which overwrites the receiver, so a still-needed multiplicand is copied first. The register blocks are therefore the largest that do not spill (2×4 for AVX2 and 3×4 for AVX-512), and each fused multiply-add makes an operand at its last use the receiver where one exists. The kernels do not use `archsimd.LoadFloat32x8Part` or `LoadFloat32x16Part`, which read a full vector past the end of a short slice and could fault past the last embedded weight. A final partial chunk is instead loaded as the slice's last full vector, with the lanes already counted zeroed by a mask; only a slice shorter than one vector is copied into a zeroed local array. Copying every tail cost about a third of the matrix product's throughput at an odd reduction length.

On amd64, Go compiles scalar float32 and float64 arithmetic to legacy SSE instructions, which run several times slower on Intel CPUs while the upper bits of the AVX registers hold data. Go 1.27 does not clear them, so every vector kernel executes `VZEROUPPER` (`archsimd.ClearAVXUpperBits`) before it returns or runs scalar code. That alone is not enough: under `GOEXPERIMENT=simd`, the runtime's asynchronous preemption restores every vector register in full, leaving the upper bits dirty at an arbitrary point in any goroutine, and they stay dirty until the next `VZEROUPPER`. Every scalar kernel therefore also clears them on entry, through `clearUpperBits` (`clear_upper_bits_simd_amd64.go`, and a no-op in `clear_upper_bits_select_generic.go` for every other build), which bounds the slowdown to the one kernel call a preemption interrupts. Scalar code outside the kernels, the caller's included, can still meet dirty upper bits after a preemption until the next vector instruction clears them; that is a property of the Go runtime, not of this library.

Kernel tests exercise only the exported functions, so the same tests verify both builds. Under `goexperiment.simd`, internal tests also compare each SIMD implementation the CPU supports with its generic function across shapes that reach every tail path, so an amd64 CPU with AVX-512 tests its AVX2 kernels too. Kernel benchmarks establish the baseline a SIMD build is measured against.

#### The SIMD Build

A consumer enables the SIMD kernels by building with the `GOEXPERIMENT=simd` environment variable, for example `GOEXPERIMENT=simd go build ./...`. The go command derives the `goexperiment.simd` build tag from that setting; passing the tag through `-tags` alone fails to link. The setting is toolchain-wide: it applies to every package in the binary, and the `simd/archsimd` package it enables is outside the Go 1 compatibility promise, so a Go release may change it incompatibly. Architectures without a SIMD file build with the generic kernels under either setting. The public API, the model modules, and the default build are identical with and without the experiment.

Three rules govern SIMD numerics:

1.  **Every output uses one procedure.** Within an accelerated kernel, every output element is computed with the same lane split, reduction order, and tail handling, whether it falls in a full tile, a leftover row, or a leftover column. A row's result therefore does not depend on how many rows the call holds or on how the caller splits rows across goroutines, so a forecast is identical at every parallelism limit.
2.  **Builds agree within float32 rounding, not bitwise.** The generic and SIMD builds, SIMD builds on architectures with different vector widths, and amd64 SIMD builds on CPUs at different vector levels accumulate in different orders, and the SIMD `SoftmaxRows` replaces `math.Exp` with a polynomial exponential within 2 ulp of the correctly rounded float32 result. Their results differ by float32 rounding and agree within the parity tolerances. `AddBias`, `ReLU`, and `AddBiasReLU` perform one float32 operation per element, so they agree bitwise across builds, apart from the sign and payload of a NaN. Some softmax results are exact in every build, and callers may rely on them: an exponent of zero gives exactly 1, an exponential below the float32 underflow threshold, or of negative infinity, gives exactly 0, so a masked score gets exactly zero weight, and a fully masked row gets exactly `1/cols` in every position.
3.  **Both builds pass parity.** The SIMD build meets the same tolerances as the generic build, and the same tasks appear under "Known Deviations": each listed task still deviates under the SIMD build on every checkpoint.

Parallelism sits above the kernels. The network divides work by row, head, or output tile and hands each range to a single-threaded kernel call through `internal/parallel`, bounded by the call's parallelism limit. Changing a kernel's implementation never changes the concurrency structure.

Time attention lays its keys and values out for the kernels. After the rotary embeddings, one parallel step copies each (row, head) pair's keys into a contiguous `[keys × 64]` matrix and its values into a transposed `[64 × keys]` matrix, once for all of the row's query blocks; work units of up to 64 queries then read them in place. The key count is _S_ rounded up to a multiple of 16, the widest vector, so the attention-weighted value product has no partial chunk and every score row starts on a vector boundary. Padding keys and values are zero, and the time mask gives padding keys negative infinity rather than the most negative finite value, so they receive exactly zero weight even in a row whose real keys are all masked, and the padding changes results only by the summation order of float32 rounding.

### Batching and Memory

Package `internal/batching` converts a call's tasks into engine batches:

1.  **Rows.** Each task contributes its targets first, then its past-only covariates, then its known-future covariates, with covariates in each class ordered by name. Group attention ignores row order, so this ordering affects only floating-point rounding.
2.  **Encoding.** Categorical covariates are encoded to numbers exactly as the reference encodes them, from the full supplied histories before trimming to the context window:
    - In a task with one target, each category maps to the smoothed mean of that target's observed values over the steps where the category occurs (target encoding, smoothing weight 1). A future category never seen in the past maps to the target's overall mean.
    - In a task with several targets, each category maps to its ordinal position among all distinct categories of that covariate across the whole call, sorted lexicographically. A future category never seen in the past is treated as unknown at that step: it is masked out, like a `NaN` numeric value, rather than encoded as the missing category.
    - The missing category (the empty string) is a category of its own in both encodings, in the past and the future alike. Under ordinal encoding, it takes the code after every named category.
3.  **Groups.** Each task receives its own group ID; under `WithCrossLearning`, every row shares one.
4.  **Chunking.** Tasks are packed, in order, into batches of at most `WithBatchSize` rows (default 256). A task is never split; a task with more rows than the limit forms a batch of its own. Under `WithCrossLearning`, a call whose rows exceed the limit returns `ErrCrossLearningBatchTooLarge`, since sharing information only within chunks would silently depend on chunk boundaries.
5.  **Trimming and padding.** Each row is trimmed to the context window, then the rows of each batch are left-padded with `NaN` to the batch's longest row.
6.  **Future rows.** Targets and past-only covariates get all-`NaN` future rows; known-future covariates carry their supplied values. Rows are padded with `NaN` to a whole number of output patches.

The reference fills a batch while its row count is below the limit, so its batches can overshoot; this package never exceeds the limit except for a single oversized task. Because a series' forecast does not depend on its batch-mates beyond floating-point rounding, the difference does not affect parity.

Activations dominate a call's memory. Most intermediate tensors hold about _R_ × _S_ × _D_ × 4 bytes; the feed-forward network's hidden activations are four times larger (_R_ × _S_ × `d_ff`). Attention adds score matrices per concurrently processed head: _S_ × _S_ per row in time attention, and _R_ × _R_ per token position in group attention. Chunking bounds _R_; `WithContextLength` bounds _S_.

The call's memory beyond activations also follows the batch size. The batching package plans a call up front, computing the call-wide ordinal category sets and deriving the chunk boundaries from each task's row count, then builds each batch's rows and matrices just before its engine pass. The root package runs batches in order, checks the context between them, and releases each batch after routing its forecasts, so only one batch's inputs are materialized at a time; the call holds its own inputs and the returned forecasts throughout.

### Long Horizons

Package `internal/horizon` reproduces the reference pipeline's long-horizon heuristic. When the horizon exceeds `MaxHorizon()` and `WithLongHorizon()` is set:

1.  The first pass predicts `min(ceil(horizon/16), 64)` patches as usual.
2.  Each row is expanded into nine paths, one per unrolled decile (10, 20, …, 90 percent, all native to both models). Group IDs become `group × 9 + path`, so paths share information only with matching paths of the same task.
3.  Each round takes the previous round's forecast at the native levels (the first pass's, then each round's combined result from step 4), and appends its decile for each path to that path's context: path 1 receives the 10th percentile, path 9 the 90th. Contexts are then trimmed to the model's full `ContextLength()`, not to a shorter `WithContextLength` window, so they may grow during unrolling. Where a known-future covariate has a value, the covariate's actual value replaces its predicted value. The round then predicts the next chunk of up to 64 patches.
4.  Each round's 9 × _Q_ predictions per step are reduced to the native quantile levels with a weighted quantile. The weights are the outer product of each level set's probability mass, where level _i_ receives mass proportional to (_q_<sub>_i_+1</sub> − _q_<sub>_i_−1</sub>) / 2, with _q_<sub>0</sub> = 0 and _q_<sub>_n_+1</sub> = 1.
5.  Rounds repeat until the horizon is covered; the result is truncated to the requested horizon.

Each extra round costs about nine forward passes over the original rows.

### Percentile Math

Package `internal/quantile` provides the two interpolations the reference uses:

- **Interpolation across levels:** for a query level between two native levels, linear interpolation of their values. `Forecast.Percentile` uses it, as does the long-horizon heuristic when extracting deciles.
- **Weighted quantile:** samples are sorted, their normalized weights accumulated into an empirical CDF, and query levels interpolated against that CDF. Below the first cumulative weight, the result is the lowest sample; above the last, if rounding leaves it short of 1, the highest.

Public percentages are converted to the model's quantile levels (`p / 100`) at the API boundary.


## Verification

### Test Tiers

[`AGENTS.md`](../AGENTS.md) describes the test tiers and the `make` targets that run them. Every tier runs in both the default and the SIMD build. No test touches the network or an external system, so the project has no boundary or blackbox suites.

### Parity Fixtures

Parity fixtures are generated once from the reference implementation, `chronos-forecasting` v2.3.2 on PyTorch in fp32 on CPU, and committed as safetensors files with a JSON manifest per scenario recording its inputs, options, and generating versions. No fixture-generation code lives in the repository.

- **Intermediate fixtures** capture one short-history forward pass per model layer by layer: normalized context, patch inputs, input embeddings, each encoder layer's output, the final hidden states, and pre-denormalization quantiles. They localize a mismatch to one stage.
- **Synthetic-checkpoint fixtures** come from a tiny Chronos-2 checkpoint (a saved copy of a model, here randomly initialized rather than trained) with two layers of width 32, committed in the root module's `testdata/`, so the root module tests the full network without real weights.
- **Scenario fixtures** cover, for each model: a short univariate history; a history with gaps; histories whose lengths are and are not multiples of 16; a constant series; a batch of different-length histories; a multivariate task; numeric past-only and known-future covariates; categorical covariates under both target and ordinal encoding; cross-learning; a history longer than 8192 steps; a 1024-step horizon; a long-horizon forecast; and embeddings.

Real-model fixtures and the tests that read them live together in the `parity` module, under `testdata/base/` and `testdata/small/`, so no published module's archive carries them. The parity tests may import the root module's internal packages, because Go's `internal` rule compares import paths and `parity` shares the root's path prefix. The synthetic-checkpoint fixtures stay in the root module: they amount to tens of kilobytes, and nothing outside a test reads them, so they never reach a consumer's binary.

Comparisons use combined tolerances, `|got − want| ≤ abs + rel·|want|`, with parameters:

| Parameter | Default | Applies To |
|---|---|---|
| `stageAbsTolerance`, `stageRelTolerance` | 1e-4, 1e-4 | intermediate tensors |
| `forecastAbsTolerance`, `forecastRelTolerance` | 1e-4 × the row's scale, 1e-3 | final quantiles |

#### Known Deviations

A known deviation is a fixture task whose reference output chronos2-go deliberately does not reproduce. Each harness lists its known deviations explicitly, matching this table. For those tasks, parity checks only that the forecast has the reference's shape and finite values and, for trained checkpoints, that quantile levels never decrease from one level to the next; the synthetic checkpoint's random weights do not order its levels even in the reference. An exempt task that matches the reference within tolerance fails, so the list cannot go stale. Every other task is compared at full tolerance.

| Checkpoint | Scenario | Task | Cause |
|---|---|---|---|
| synthetic | `categorical_target_encoding` | 1 | Constant row (see "How Chronos-2 Works") |
| `chronos-2-small` | `categorical_target_encoding` | 1 | Constant row (see "How Chronos-2 Works") |
| `chronos-2` | `categorical_target_encoding` | 1 | Constant row (see "How Chronos-2 Works") |

### Benchmarks

Kernel benchmarks in `internal/kernel` cover each kernel at the model's real dimensions. Benchmarks in `parity` cover the network's forward pass and end-to-end forecasts, long-horizon forecasts, and embeddings for each model, with the base model's behind the `slow` build tag. Running them in both builds measures the SIMD kernels against the generic ones.
