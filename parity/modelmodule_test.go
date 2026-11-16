package parity

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"testing"
)

// modelModulePaths maps each real model's short name, as the fixtures under
// testdata/ and the test names use it, to the path of the module that embeds
// its checkpoint.
var modelModulePaths = map[string]string{
	"base":  "github.com/pinkhop/chronos2-go-base",
	"small": "github.com/pinkhop/chronos2-go-small",
}

// errModuleNotDownloaded reports that the go command knows a module but has
// no copy of its source on disk.
var errModuleNotDownloaded = errors.New("module source is not downloaded")

// modelModuleDir returns the directory holding the source of the named real
// model's module, "small" or "base". The directory is the module's checkout
// when the active workspace uses it, and its read-only copy in the module
// cache otherwise. It fails tb when the name is unknown or the go command
// cannot locate the module.
func modelModuleDir(tb testing.TB, model string) string {
	tb.Helper()
	modulePath, ok := modelModulePaths[model]
	if !ok {
		tb.Fatalf("no model module is named %q", model)
	}
	dir, err := moduleDir(tb.Context(), modulePath)
	if err != nil {
		tb.Fatalf("locate the %s model's module: %v", model, err)
	}
	return dir
}

// moduleDir asks the go command, run from the current directory, where the
// source of the module with modulePath lies. It runs `go list -m`, so the
// answer follows the same workspace, replace directives, and module cache
// that builds of this module use. It returns an error wrapping
// errModuleNotDownloaded when the module's source is not on disk, and an
// error carrying the go command's diagnostics when the command fails.
// Canceling ctx stops the command.
func moduleDir(ctx context.Context, modulePath string) (string, error) {
	// #nosec G204 -- fixed go subcommand; modulePath comes from modelModulePaths.
	command := exec.CommandContext(ctx, "go", "list", "-m", "-f", "{{.Dir}}", modulePath)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return "", fmt.Errorf("go list -m %s: %w: %s", modulePath, err, strings.TrimSpace(stderr.String()))
	}
	dir := strings.TrimSpace(string(output))
	if dir == "" {
		return "", fmt.Errorf("go list -m %s: %w", modulePath, errModuleNotDownloaded)
	}
	return dir, nil
}
