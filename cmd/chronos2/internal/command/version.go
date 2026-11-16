package command

import (
	"context"
	"fmt"
	"runtime/debug"

	"github.com/urfave/cli/v3"
)

// unknownVersion is printed when the binary carries no build information.
const unknownVersion = "unknown"

// versionCommand returns the version subcommand, which prints the module
// version and VCS revision from the binary's build information.
func versionCommand(buildInfo func() (*debug.BuildInfo, bool)) *cli.Command {
	return &cli.Command{
		Name:  "version",
		Usage: "print the version and VCS revision",
		Action: func(_ context.Context, cmd *cli.Command) error {
			if _, err := fmt.Fprint(cmd.Root().Writer, formatVersion(buildInfo())); err != nil {
				return fmt.Errorf("write output: %w", err)
			}
			return nil
		},
	}
}

// formatVersion renders the version report: "chronos2 <version>", then, when
// the build recorded one, "revision <hash>" marked "(modified)" for a dirty
// tree. ok is false when info is unavailable, as debug.ReadBuildInfo reports.
func formatVersion(info *debug.BuildInfo, ok bool) string {
	if !ok || info == nil {
		return "chronos2 " + unknownVersion + "\n"
	}
	version := info.Main.Version
	if version == "" {
		version = unknownVersion
	}
	report := "chronos2 " + version + "\n"

	var revision string
	var modified bool
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if revision == "" {
		return report
	}
	report += "revision " + revision
	if modified {
		report += " (modified)"
	}
	return report + "\n"
}
