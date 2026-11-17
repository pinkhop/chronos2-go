# Agent Guide

This repository is `chronos2-go`, a pure-Go library that runs Amazon's Chronos-2 time series forecasting model on-device, with the model weights embedded in the consumer's binary. The weights live in sibling repositories: one per model, plus four that hold the base model's shards. These instructions apply to any agent or developer changing the code in this repository.

Read [`docs/DESIGN.md`](docs/DESIGN.md) before changing behavior. It is the source of truth for packaging, internal architecture, and verification, and it states the principles the public API follows. The doc comments of package `chronos2` are the source of truth for the public API itself. When a change alters the design or the public contract, update `docs/DESIGN.md` or the doc comments in the same change.


## Repository Layout

The repository holds two Go modules:

| Directory | Module Role |
|---|---|
| `.` | Package `chronos2` (the public API) and every `internal/` package. |
| `parity` | Real-model parity tests and their fixtures, the weights-manifest and embed-aliasing checks on the model modules, and the module-size check. Never released. |

Each weight module is the only module in a repository of its own, at that repository's root:

| Repository | Module Role |
|---|---|
| [`github.com/pinkhop/chronos2-go-base`](https://github.com/pinkhop/chronos2-go-base) | Package `base`: embeds the `amazon/chronos-2` config and assembles the shards of the four part modules. |
| [`github.com/pinkhop/chronos2-go-base-part1`](https://github.com/pinkhop/chronos2-go-base-part1) through [`-part4`](https://github.com/pinkhop/chronos2-go-base-part4) | Packages `part1` through `part4`: each embeds three of the base checkpoint's 12 shards and returns them through `Shards()`. |
| [`github.com/pinkhop/chronos2-go-small`](https://github.com/pinkhop/chronos2-go-small) | Package `small`: embeds the `autogluon/chronos-2-small` checkpoint. |

A weight repository holds only its Go source, `weights/`, `LICENSE`, `NOTICE`, `README.md`, `AGENTS.md`, a `Makefile`, and the `mise` files. Its `AGENTS.md` governs changes there.

Inside the root module:

- `internal/kernel`: numeric primitives, the layer the SIMD build accelerates.
- `internal/parallel`: bounded fan-out of index ranges across goroutines.
- `internal/network`: the Chronos-2 forward pass over an engine batch.
- `internal/batching`: turns tasks into engine batches.
- `internal/quantile`: percentile interpolation and weighted quantiles.
- `internal/horizon`: long-horizon autoregressive unrolling.
- `internal/safetensors`: safetensors parsing and typed tensor views.
- `internal/testmodel`: test support shared across packages: the synthetic checkpoint, fixture and scenario readers, and the comparisons parity tests use.

The root module's `testdata/synthetic/` holds a tiny randomly initialized checkpoint (two layers, width 32) and reference fixtures for it, so the root module tests the full network without real weights.


## Commands

Tool versions, including Go itself, are pinned in `mise.toml` and `mise.lock`. Run `mise install` once, then use the `Makefile`, which runs each target across every module in this repository:

| Command | Purpose |
|---|---|
| `make help` | List all targets. |
| `make format` | Format Go files with `goimports`, then `gofumpt`. |
| `make lint` | Check formatting, then run `go vet`, `staticcheck`, `errcheck`, `ineffassign`, `exhaustive`, `misspell`, and `typos`. |
| `make security` | Run `gosec` and `govulncheck`. |
| `make test` | Run the default test tier under the race detector (alias for `make test-units`). |
| `make test-slow` | Run only the tests behind the `slow` build tag, without the race detector. |
| `make test-simd` | Run the default test tier under the race detector in the SIMD build (`GOEXPERIMENT=simd`). |
| `make test-slow-simd` | Run only the `slow` tier in the SIMD build. |
| `make test-all` | Run both tiers in both builds. |
| `make clean` | Remove the test cache. |

`make lint` and `make security` run their type-checked analyzers in both builds, so a single run covers the SIMD build too.

Before considering a change complete, run `make format`, `make lint`, `make security`, and `make test`. Run `make test-simd` as well whenever touching `internal/kernel`. Run `make test-all` when touching the network, kernels, batching, horizon logic, weights, or fixtures. The test targets test every module even after one fails, then fail if any did.

Benchmarks are not part of any target. Kernel benchmarks live in `internal/kernel`; end-to-end forecast and embedding benchmarks live in `parity`, with the base model's behind the `slow` tag. For example, from `parity/`:

```console
$ go test -tags slow -run '^$' -bench 'Forecast$' -benchmem .
```

Add `-cpu 1` to measure a single thread.

`parity` takes the model modules from the module cache at the versions its `go.mod` requires, so its real-model tests run against the published weights; the first run downloads both models.


## Hard Constraints

These protect properties that are expensive or impossible to restore once broken.

### Purity and Portability

- No cgo. No network or file-system access at run time; checkpoints arrive only as embedded data.
- The library never logs and never writes to standard streams. It reports problems by returning errors.
- `unsafe` appears only where `Load` aliases embedded weight bytes, with a comment justifying it.
- Add third-party dependencies only to `parity`, and only after discussing the need. The root module depends on the standard library alone (plus test-only dependencies such as `github.com/google/go-cmp`), and the model modules on the root module alone.

### Weights and Modules

- The rules for weights and model modules (no Git LFS, the shard-size and manifest rules, `[]byte` embeds, and no tests in a model module) live in the `AGENTS.md` of each weight repository. Follow them whenever a change reaches those repositories.
- `parity` enforces part of them from here. Its embed test checks that every package embedding shards (`small` and the four base parts) embeds them as `[]byte`, and its alias test checks that `Load` views them without copying (it aliases aligned shard bytes as float32 tensors, and the Go linker currently aligns `[]byte` embeds but not string data). Its manifest test checks each model's config and shards against the `weights/manifest.json` of the modules holding them, and against the upstream checkpoint. Keep all three passing.
- Never commit weights or other large files to this repository. Go downloads the root module as an archive of the whole repository, nested modules included, and refuses that archive above 500 MiB.
- No change may make this repository's whole-repository archive, or any module it publishes, exceed the Go module limits. The module-size test in `parity`, part of `make test`, checks both.
- No module may require `parity`. Only `parity` imports `parity`.

### Parity Fixtures

- Fixtures in `parity/testdata/` and `testdata/` were generated from the reference implementation (`chronos-forecasting` v2.3.2, PyTorch fp32 on CPU). Treat them as ground truth.
- Never edit or regenerate a fixture, or loosen a tolerance, to make a failing test pass. A parity failure means the Go code diverges from the reference; find the divergence.
- Tolerances are defined in `docs/DESIGN.md`. Change them only by changing the design.
- The only fixture tasks exempt from comparison are those listed under "Known Deviations" in `docs/DESIGN.md`, and each parity harness lists exactly the same tasks. Adding an exemption is a design change. An exempt task that starts matching the reference fails its test; remove the exemption from both places when that happens.

### Numerics

- Reproduce the reference model's behavior exactly, including its architectural quirks: unscaled attention scores, additive masks using the most negative finite float32, T5-style RMS norms, and the rotate-half rotary embedding. `docs/DESIGN.md` lists them all.
- Do not reproduce accidents of PyTorch's floating-point evaluation, such as its summation order. Where correct arithmetic and the reference disagree visibly, keep the correct result and document the difference under "Known Deviations".
- Compute in float32. The public API accepts and returns float64 and converts at the boundary.

### Kernels

Everything in `internal/kernel` follows one contract, so a SIMD implementation can replace any function without touching callers:

- Operate on flat `[]float32` slices with explicit dimensions.
- Write into caller-provided output slices; never allocate.
- Never start goroutines; parallelism belongs to the caller through `internal/parallel`.
- Keep each operation in its own file with a `_generic.go` suffix and no build constraint. An operation with a SIMD implementation uses the file layout in `docs/DESIGN.md` ("Kernels and SIMD Seams"): the exported function and its length checks in an unsuffixed file, the always-compiled generic code in `_generic.go`, a `_select_generic.go` selector for builds without SIMD, and, per architecture, files ending in `_simd_<arch>.go` constrained to `goexperiment.simd`.
- Select the SIMD build with the `GOEXPERIMENT=simd` environment variable, never with `-tags goexperiment.simd`, which fails to link.
- Compute every output of an accelerated kernel with one procedure (lane split, reduction order, and tail), so a row's result never depends on the number of rows or how the caller splits them.
- Expect the generic and SIMD builds, and SIMD builds on different architectures, to agree within float32 rounding, not bitwise. Both builds must pass parity at the same tolerances, with the same Known Deviations.
- Test and benchmark only through the exported functions, so one suite verifies every implementation. The one exception is the SIMD-only internal test that compares each SIMD implementation with its generic function directly.

### Public API

- Keep the univariate path minimal: `m.Forecast(ctx, history, horizon)` must never require more than that.
- Add capabilities as new methods or options. Never change an existing signature.
- The `F32` suffix is reserved for a possible future float32 surface (`ForecastF32` and similar). Do not use it for anything else.
- A `*Model`, a `Forecast`, and an `Embedding` are immutable. Accessors return copies.
- Every returned error matches a sentinel declared in `errors.go` under `errors.Is`.


## Go Conventions

### Language Version

The modules target Go 1.27.4. Prefer the modern standard library and language forms over older idioms: `for i := range n`; `min`, `max`, and `clear`; the `slices`, `maps`, and `iter` packages; `errors.Join` and `errors.AsType`; `sync.OnceValue` and `sync.OnceValues`; `sync.WaitGroup.Go`; `context.WithoutCancel` and `context.AfterFunc`; `new(expr)`; struct literals naming promoted fields; and, in tests, `t.Context()` and `b.Loop()`. `go vet` reports standard-library symbols newer than a file's Go version; treat those reports as errors.

### Errors

- Assign every returned error, including one being discarded: write `_ = f()` or `_, _ = fmt.Fprintf(...)`, never a bare call. A discarded error needs a comment explaining why discarding is safe.
- Handle each error: wrap it with context using `%w` and return it, or act on it. Discard only best-effort errors, and only at the highest level that can judge them.
- When returning an error from a lower layer, wrap it so the caller sees the kind of failure, not implementation details.
- Compare errors with `errors.Is` or `errors.AsType`; never match on error text.

### Naming and Packages

- Prefer descriptive names that read clearly at the call site, even when longer than typical Go. Conventional short names remain fine for receivers, loop indices, and locals such as `ctx`, `err`, and `t`.
- Never prefix a symbol with its package's name (`kernel.MatMulTransposed`, not `kernel.KernelMatMul`).
- Name packages for what they provide. Never create `util`, `utils`, `helpers`, or `common` packages.
- Define interfaces in the package that consumes them, not the one that implements them.
- Enumerations use `iota` with an explicit zero value meaning "unspecified".
- Use American English in identifiers, file and directory names, and comments.

### Structure

- Prefer many small, focused files: typically 200–400 lines, never more than 800.
- Keep functions under 50 lines and nesting at most four levels deep.
- Use named constants instead of literal values whose meaning is not obvious.
- Prefer immutable data: functions that would modify shared state return a new value instead.

### Concurrency

- No goroutine outlives the call that started it.
- Honor `ctx` cancellation between units of work.
- In channel pipelines, producers own cancellation and close their output; consumers drain their input until it closes. Channels are unbuffered unless a specific buffer size can be justified.

### Comments

- Every declaration — exported or not, including types, functions, methods, constants, variables, and struct fields — has a doc comment.
- Doc comments state what the declaration does and why a caller would use it. Document parameters (units, valid ranges, and whether `nil` is accepted), return values, errors the caller must handle, side effects, and concurrency guarantees.
- Inside functions, use brief comments to mark the steps of a multi-step operation, and otherwise comment only to explain why: workarounds, numerical subtleties, and deliberate deviations.
- Write actionable TODOs as `// TODO(owner): description`. Never leave placeholder comments.
- Use American spelling and the serial comma ("the API, database, and cache").


## Testing

### Approach

- Write tests first: red, then green.
- Structure every test as Given/When/Then, with exactly one When. A failing Given step calls `t.Fatalf` so setup failures are distinguishable from failures of the behavior under test.
- Then blocks only assert on values captured during When. Never call a method from the When block inside Then. To verify a second behavior, write a second test.
- Never use mocks or mock generators. Use hand-written stubs, fakes, spies, and dummies, and keep shared fakes in dedicated packages such as `internal/testmodel`.
- Use the standard `testing` package only, plus `github.com/google/go-cmp` for comparing complex values. No assertion libraries.

### Conventions

- Keep tests in the package under test, focused on behavior observable through exported functions; add internal tests only where complex logic warrants them.
- Name tests `TestTypeName_Scenario_ExpectedOutcome` or `TestFunctionName_Scenario_ExpectedOutcome`.
- Call `t.Parallel()` in unit tests and their table-driven subtests.
- Store static fixtures under `testdata/`.
- Write benchmarks with `for b.Loop()`.

### Tiers

- **Default:** unit and collaboration tests, the module-size check, and small-model parity on short histories. `make test` runs these under the race detector; they must stay fast enough to run before every change.
- **Slow:** base-model parity, full 8192-step contexts, long-horizon unrolling, and the remaining parity scenarios. Guard these files with `//go:build slow`; `make test-all` runs them.
- **SIMD:** both tiers again with `GOEXPERIMENT=simd`, through `make test-simd` and `make test-slow-simd`; `make test-all` runs them too.


## Tooling

- Pin every Go-based developer tool in `mise.toml` (adding with `mise use --pin`) and commit `mise.lock`. Never declare tools with `go.mod` `tool` directives or run them with `go tool`.
- Never use `golangci-lint`; the `Makefile` runs each analyzer directly.
- `go-errorlint` is pinned but not run, because its pinned release cannot load packages compiled by Go 1.27. Add it to `make lint` once a release can.
- Keep the Go version in `mise.toml` and every `go.mod` in sync, including the weight repositories'.
- `simd/archsimd`, which the SIMD build uses, is outside the Go 1 compatibility promise and may change incompatibly in any Go release. Upgrade Go only after `make test-simd` and `make lint` pass on the new version.


## Documentation

- Documents describe the current state of the project. Keep decision history, superseded approaches, and change narratives out of them; commit messages carry history.
- Markdown uses ATX headings in title case, two blank lines before every H1 and H2 (except a first-line title), `-` list markers, four-space nested indents, one paragraph per source line, and a language tag on every fenced code block.


## Commits

Write commit messages in the Conventional Commits format (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`), with the subject in lowercase imperative mood.
