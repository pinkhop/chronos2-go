# chronos2-go — formatting, analysis, and tests across every module.
#
# `make help` lists every public target alphabetically with its one-line
# description. The listing is generated from the `## target: description`
# comments below, so a new target documents itself by carrying one.
#
# The repository holds two Go modules.
# `go vet ./...` and its relatives stop at a nested go.mod, so every
# type-checked target loops over MODULES and runs inside each module directory.
#
# Every target is phony; the Go toolchain already does its own staleness
# tracking. Every tool a target runs comes from PATH. The repository's mise.toml
# pins each one, together with the Go toolchain, and `mise install` provisions
# them all.


###############################################################################
# MODULE SETTINGS

# Every module in the repository, root first.
MODULES       := . parity

# The build-tag sets the type-checked analyses run under, one pass per set; the
# literal "untagged" names the default set, which has no -tags flag.
ANALYSIS_TAG_SETS := untagged slow

# The GOEXPERIMENT settings the type-checked analyses run under, one pass per
# setting and tag set; the literal "default" keeps the caller's own setting.
ANALYSIS_EXPERIMENTS := default simd

# The GOEXPERIMENT value that selects the SIMD kernels. The go command derives
# the goexperiment.simd build tag from it; passing that tag through -tags alone
# fails to link, so SIMD targets set the environment variable instead.
SIMD_EXPERIMENT := simd

# Git pathspecs that exclude generated Go code from formatting and spelling.
GENERATED_PATHSPECS :=


###############################################################################
# COMMON SETTINGS

# Every hand-written Go file Git does not ignore, including new files not yet
# added, except vendored code, test fixtures, and generated code.
GO_FILES       = git ls-files --cached --others --exclude-standard '*.go' \
	':(exclude,glob)**/vendor/**' ':(exclude,glob)**/testdata/**' \
	':(exclude,glob).worktrees/**' $(GENERATED_PATHSPECS)

# The slow tier runs full-size models over long contexts without the race
# detector, which would multiply its run time many times over.
SLOW_TEST_FLAGS := -tags=slow -timeout=120m

.DEFAULT_GOAL := help
.PHONY: clean format help lint security security-gosec security-govulncheck test test-all test-simd test-slow test-slow-simd test-units


###############################################################################
# TARGETS

## clean: remove the test cache
clean:
	go clean -testcache

## format: rewrite the hand-written Go files with goimports, then gofumpt
format:
	@files=$$($(GO_FILES)); \
	if [ -n "$$files" ]; then goimports -w $$files && gofumpt -w $$files; fi

## help: list the available targets
help:
	@echo "chronos2-go — available targets:"
	@echo
	@grep -hE '^## [a-z0-9-]+:' $(MAKEFILE_LIST) \
		| sed -e 's/^## //' \
		| awk -F': ' '{ printf "%s\t%s\n", $$1, $$2 }' \
		| sort \
		| awk -F'\t' '{ printf "  %-22s %s\n", $$1, $$2 }'

## lint: check formatting, run the analyzers per experiment, module, and build-tag set, then check spelling
#
# The formatters run in check mode, so lint never mutates the tree. go vet,
# staticcheck, and errcheck take a -tags flag; ineffassign and exhaustive have
# none and read the tags from GOFLAGS instead. Every analyzer reads GOEXPERIMENT
# from the environment. errcheck's -blank is deliberately not set, because a
# justified discard is written `_ = f()`.
#
# go-errorlint is pinned but not run: v1.8.0 cannot load packages compiled by
# Go 1.27. Add `GOFLAGS="$$goflags" go-errorlint -errorf ./...` to the loop once
# a release can.
lint:
	@files=$$($(GO_FILES)); \
	for formatter in goimports gofumpt; do \
		echo "==> $$formatter"; \
		offenders=$$([ -z "$$files" ] || $$formatter -l $$files); \
		if [ -n "$$offenders" ]; then \
			echo "not $$formatter-formatted:"; echo "$$offenders"; \
			echo "run: make format"; exit 1; \
		fi; \
	done
	@for experiment in $(ANALYSIS_EXPERIMENTS); do \
		if [ "$$experiment" = "default" ]; then goexperiment="$$GOEXPERIMENT"; \
		else goexperiment="$$experiment"; fi; \
		for module in $(MODULES); do \
			for tags in $(ANALYSIS_TAG_SETS); do \
				if [ "$$tags" = "untagged" ]; then flag=""; goflags="$$GOFLAGS"; \
				else flag="-tags=$$tags"; goflags="$$GOFLAGS -tags=$$tags"; fi; \
				pass="$$tags, $$experiment experiment"; \
				echo "==> $$module: go vet ($$pass)"; (cd $$module && GOEXPERIMENT="$$goexperiment" go vet $$flag ./...) || exit 1; \
				echo "==> $$module: staticcheck ($$pass)"; (cd $$module && GOEXPERIMENT="$$goexperiment" staticcheck $$flag ./...) || exit 1; \
				echo "==> $$module: errcheck ($$pass)"; (cd $$module && GOEXPERIMENT="$$goexperiment" errcheck -asserts $$flag ./...) || exit 1; \
				echo "==> $$module: ineffassign ($$pass)"; (cd $$module && GOEXPERIMENT="$$goexperiment" GOFLAGS="$$goflags" ineffassign ./...) || exit 1; \
				echo "==> $$module: exhaustive ($$pass)"; (cd $$module && GOEXPERIMENT="$$goexperiment" GOFLAGS="$$goflags" exhaustive -d ./...) || exit 1; \
			done; \
		done; \
	done
	@echo "==> misspell"
	@files=$$($(GO_FILES)); [ -z "$$files" ] || misspell -locale US -source=go -error $$files
	@echo "==> typos"
	@typos --locale en-us

## security: run every security scanner — gosec, then govulncheck
security: security-gosec security-govulncheck

## security-gosec: scan each module for insecure code patterns per experiment and build-tag set
security-gosec:
	@for experiment in $(ANALYSIS_EXPERIMENTS); do \
		if [ "$$experiment" = "default" ]; then goexperiment="$$GOEXPERIMENT"; \
		else goexperiment="$$experiment"; fi; \
		for module in $(MODULES); do \
			for tags in $(ANALYSIS_TAG_SETS); do \
				if [ "$$tags" = "untagged" ]; then flag=""; else flag="-tags=$$tags"; fi; \
				echo "==> $$module: gosec ($$tags, $$experiment experiment)"; \
				(cd $$module && GOEXPERIMENT="$$goexperiment" gosec -quiet -exclude-generated $$flag ./...) || exit 1; \
			done; \
		done; \
	done

## security-govulncheck: check each module's dependencies against the Go vulnerability database
#
# Not hermetic: it needs network access and consults a database that changes
# independently of this repository.
security-govulncheck:
	@for module in $(MODULES); do \
		echo "==> $$module: govulncheck"; \
		(cd $$module && govulncheck ./...) || exit 1; \
	done

## test: run the default (fast) tier (alias for test-units)
test: test-units
	@echo "default tier only; run 'make test-all' to add the slow tier and the SIMD build"

## test-all: run every tier in both builds — default, slow, SIMD default, then SIMD slow
test-all: test-units test-slow test-simd test-slow-simd

## test-simd: run the default tier in every module under the race detector with the SIMD kernels
#
# Like test-units, it tests every module before failing.
test-simd:
	@status=0; for module in $(MODULES); do \
		echo "==> $$module: go test ($(SIMD_EXPERIMENT) experiment)"; \
		(cd $$module && GOEXPERIMENT=$(SIMD_EXPERIMENT) go test -race ./...) || status=1; \
	done; exit $$status

## test-slow: run the slow tier — base-model parity, full contexts, and long horizons
#
# Like test-units, it tests every module before failing.
test-slow:
	@status=0; for module in $(MODULES); do \
		echo "==> $$module: go test (slow)"; \
		(cd $$module && go test $(SLOW_TEST_FLAGS) ./...) || status=1; \
	done; exit $$status

## test-slow-simd: run the slow tier with the SIMD kernels
#
# Like test-units, it tests every module before failing.
test-slow-simd:
	@status=0; for module in $(MODULES); do \
		echo "==> $$module: go test (slow, $(SIMD_EXPERIMENT) experiment)"; \
		(cd $$module && GOEXPERIMENT=$(SIMD_EXPERIMENT) go test $(SLOW_TEST_FLAGS) ./...) || status=1; \
	done; exit $$status

## test-units: run the default tier in every module under the race detector
#
# Every module is tested even after one fails, so a single run reports every
# module's failures; the target still fails if any module did.
test-units:
	@status=0; for module in $(MODULES); do \
		echo "==> $$module: go test"; \
		(cd $$module && go test -race ./...) || status=1; \
	done; exit $$status
