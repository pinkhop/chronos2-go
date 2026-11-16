package main

import (
	"errors"
	"fmt"

	chronos2 "github.com/pinkhop/chronos2-go"
)

// withFlagHints adds the command-line remedy to a library error whose message
// names a Go option, so a terminal user learns which flag to pass. The result
// still matches every sentinel err matches; errors with no flag remedy are
// returned unchanged.
func withFlagHints(err error) error {
	if errors.Is(err, chronos2.ErrHorizonTooLong) {
		return fmt.Errorf("%w (on the command line, pass --long-horizon)", err)
	}
	return err
}
