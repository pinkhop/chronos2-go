package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// TestWithFlagHints_HorizonTooLong_AddsLongHorizonFlagHint verifies that
// withFlagHints points a horizon-too-long error at the --long-horizon flag
// while keeping the sentinel.
func TestWithFlagHints_HorizonTooLong_AddsLongHorizonFlagHint(t *testing.T) {
	t.Parallel()

	// Given a library error wrapping ErrHorizonTooLong
	err := fmt.Errorf("%w: 1100 steps exceed 1024", chronos2.ErrHorizonTooLong)

	// When flag hints are added
	hinted := withFlagHints(err)

	// Then the message names the flag and the sentinel still matches
	if !strings.Contains(hinted.Error(), "--long-horizon") || !errors.Is(hinted, chronos2.ErrHorizonTooLong) {
		t.Errorf("hinted = %v, want a --long-horizon hint wrapping ErrHorizonTooLong", hinted)
	}
}

// TestWithFlagHints_OtherError_ReturnsItUnchanged verifies that withFlagHints
// leaves errors without a flag remedy untouched.
func TestWithFlagHints_OtherError_ReturnsItUnchanged(t *testing.T) {
	t.Parallel()

	// Given an error unrelated to any flag
	err := fmt.Errorf("history: %w", chronos2.ErrEmptyHistory)

	// When flag hints are added
	hinted := withFlagHints(err)

	// Then the same error comes back
	if hinted != err {
		t.Errorf("hinted = %v, want %v unchanged", hinted, err)
	}
}
