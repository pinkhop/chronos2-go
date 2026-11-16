package parity

import (
	"archive/zip"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/mod/module"
	modzip "golang.org/x/mod/zip"
)

// publishedModules lists every module the repository releases, by module
// path and directory relative to the repository root. The parity module is
// absent because it is never published.
var publishedModules = []struct {
	// path is the module path.
	path string
	// dir is the module's directory relative to the repository root.
	dir string
}{
	{path: "github.com/pinkhop/chronos2-go", dir: ""},
	{path: "github.com/pinkhop/chronos2-go/cmd/chronos2", dir: "cmd/chronos2"},
}

// moduleSizeHeadroomBytes is the minimum room each published module must
// leave below the module proxy's 500 MB uncompressed limit, so that a modest
// change to a module fails here rather than at publication.
const moduleSizeHeadroomBytes = 10_000_000

// maxRepositoryArchiveBytes is the largest raw `git archive` zip the go
// command accepts when it downloads a module from a Git repository
// (codehost.MaxZipFile in cmd/go, 500 MiB). For a module at the repository
// root, the go command archives the whole repository, including every nested
// module, and applies this limit to that raw archive before it drops the
// nested modules' files. A repository that holds large nested modules can
// therefore make its root module undownloadable even when the root module's
// own files are small.
const maxRepositoryArchiveBytes = 524_288_000

// TestPublishedModules_CommittedContents_FitTheModuleProxyLimits packages each
// published module from the committed HEAD revision exactly as the module
// proxy would, with golang.org/x/mod/zip, and fails when a module violates the
// proxy's limits or comes within moduleSizeHeadroomBytes of the size limit.
func TestPublishedModules_CommittedContents_FitTheModuleProxyLimits(t *testing.T) {
	t.Parallel()

	// Given: the repository root, which holds the Git metadata the module
	// archives are built from.
	repoRoot, err := filepath.Abs("..")
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Fatalf("git is required to package modules from the committed revision: %v", err)
	}

	for _, published := range publishedModules {
		t.Run(published.dir, func(t *testing.T) {
			t.Parallel()

			// When: the module is packaged from HEAD.
			version := module.Version{Path: published.path, Version: "v1.0.0"}
			uncompressed, err := packageModule(t.TempDir(), repoRoot, version, published.dir)
			// Then: packaging succeeds within the proxy's limits, with headroom.
			if err != nil {
				t.Fatalf("module %s does not package within the module proxy's limits: %v", published.path, err)
			}
			if limit := int64(modzip.MaxZipFile) - moduleSizeHeadroomBytes; uncompressed > limit {
				t.Errorf("module %s is %d bytes uncompressed; it must stay below %d (the %d-byte limit less %d bytes of headroom)",
					published.path, uncompressed, limit, modzip.MaxZipFile, moduleSizeHeadroomBytes)
			}
			t.Logf("module %s: %d bytes uncompressed", published.path, uncompressed)
		})
	}
}

// TestRootModule_RepositoryArchive_FitsTheDownloadLimit archives the whole
// repository at the committed HEAD revision, as the go command does when it
// downloads the root module, and fails when that raw archive exceeds
// maxRepositoryArchiveBytes.
func TestRootModule_RepositoryArchive_FitsTheDownloadLimit(t *testing.T) {
	t.Parallel()

	// Given: the repository root, which holds the Git metadata the archive is
	// built from.
	repoRoot, err := filepath.Abs("..")
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Fatalf("git is required to archive the committed revision: %v", err)
	}

	// When: the whole repository is archived from HEAD.
	size, err := repositoryArchiveSize(t.TempDir(), repoRoot)
	// Then: the raw archive fits the go command's download limit.
	if err != nil {
		t.Fatalf("archive the repository: %v", err)
	}
	if size > maxRepositoryArchiveBytes {
		t.Errorf("the repository archive at HEAD is %d bytes; the go command refuses root-module downloads over %d bytes", size, maxRepositoryArchiveBytes)
	}
	t.Logf("repository archive at HEAD: %d bytes", size)
}

// repositoryArchiveSize archives the whole repository at HEAD into
// scratchDir with the same `git archive` invocation the go command uses for
// a root module, and returns the archive's size in bytes. The go command also
// disables the export-ignore and export-subst attributes; the repository sets
// neither, so the archive is the same size as the one the go command
// downloads.
func repositoryArchiveSize(scratchDir, repoRoot string) (int64, error) {
	archivePath := filepath.Join(scratchDir, "repository.zip")
	args := []string{"-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", "--prefix=prefix/", "--output=" + archivePath, "HEAD"}
	cmd := exec.Command("git", args...) // #nosec G204 -- fixed git subcommand; the only variable argument is a path in the test's temporary directory.
	cmd.Dir = repoRoot
	if output, err := cmd.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("git archive: %w: %s", err, output)
	}
	info, err := os.Stat(archivePath)
	if err != nil {
		return 0, fmt.Errorf("measure the repository archive: %w", err)
	}
	return info.Size(), nil
}

// packageModule builds the module's archive from the repository's HEAD
// revision into scratchDir, as the module proxy would, and returns the total
// uncompressed size of its files. It returns the packaging error when the
// module violates any of the proxy's limits.
//
// It lists the committed files with `git archive`, as zip.CreateFromVCS
// does, but calls zip.Create itself: CreateFromVCS refuses a Git worktree,
// whose .git is a file rather than a directory, and agents work in worktrees.
func packageModule(scratchDir, repoRoot string, version module.Version, dir string) (int64, error) {
	listing, err := committedFiles(scratchDir, repoRoot, dir)
	if err != nil {
		return 0, err
	}
	defer func() {
		// The listing was only read, so a close failure cannot affect the
		// measured size.
		_ = listing.Close()
	}()
	files := make([]modzip.File, 0, len(listing.File))
	prefix := dir + "/"
	for _, entry := range listing.File {
		name, ok := strings.CutPrefix(entry.Name, prefix)
		if dir == "" {
			name, ok = entry.Name, true
		}
		if !ok || name == "" || strings.HasSuffix(name, "/") {
			continue
		}
		files = append(files, archivedFile{name: name, entry: entry})
	}

	archivePath := filepath.Join(scratchDir, "module.zip")
	archive, err := os.Create(archivePath) // #nosec G304 -- path is inside the test's own temporary directory.
	if err != nil {
		return 0, err
	}
	if err := modzip.Create(archive, version, files); err != nil {
		_ = archive.Close() // The packaging error is the one worth reporting.
		return 0, err
	}
	if err := archive.Close(); err != nil {
		return 0, err
	}
	return uncompressedSize(archivePath)
}

// committedFiles runs `git archive` for HEAD's dir subtree and opens the
// result. The caller closes the returned reader.
func committedFiles(scratchDir, repoRoot, dir string) (*zip.ReadCloser, error) {
	listingPath := filepath.Join(scratchDir, "listing.zip")
	args := []string{"-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", "--output=" + listingPath, "HEAD"}
	if dir != "" {
		args = append(args, dir)
	} else {
		// The nested modules lie outside the root module's archive anyway;
		// leaving them out of the listing spares archiving the parity
		// fixtures.
		args = append(args, "--", ".", ":(exclude)cmd", ":(exclude)parity")
	}
	cmd := exec.Command("git", args...) // #nosec G204 -- fixed git subcommand; arguments are repository paths.
	cmd.Dir = repoRoot
	if output, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("git archive: %w: %s", err, output)
	}
	return zip.OpenReader(listingPath)
}

// uncompressedSize returns the total uncompressed size of a zip archive's
// files.
func uncompressedSize(archivePath string) (int64, error) {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return 0, err
	}
	defer func() {
		// The archive was only read, so a close failure cannot affect the
		// measured size.
		_ = reader.Close()
	}()
	var total int64
	for _, file := range reader.File {
		total += int64(file.UncompressedSize64)
	}
	return total, nil
}

// archivedFile adapts an entry of a `git archive` listing to zip.File.
type archivedFile struct {
	// name is the entry's path relative to the module root.
	name string
	// entry is the entry within the listing.
	entry *zip.File
}

// Path returns the file's path relative to the module root.
func (f archivedFile) Path() string { return f.name }

// Lstat returns the entry's file information.
func (f archivedFile) Lstat() (fs.FileInfo, error) { return f.entry.FileInfo(), nil }

// Open opens the entry's contents for reading.
func (f archivedFile) Open() (io.ReadCloser, error) { return f.entry.Open() }
