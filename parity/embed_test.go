package parity

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// embedDirective is the comment prefix that embeds a file into a variable.
const embedDirective = "//go:embed "

// shardSuffix identifies an embedded weight shard by its file name.
const shardSuffix = ".safetensors"

// TestModelPackages_EmbeddedShards_AreByteSlices verifies that every weight
// shard the model packages embed is declared as a []byte. The linker aligns
// embedded []byte data but places string data at whatever byte offset the
// layout leaves, so a string shard lets chronos2.Load view the weights in
// place only when that offset happens to be a multiple of four, and copies
// the whole checkpoint otherwise.
func TestModelPackages_EmbeddedShards_AreByteSlices(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"small", "base"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			// Given: the Go source files of every module that embeds the
			// model's shards, and a file set to parse them into.
			sources := shardModuleSources(t, model)
			files := token.NewFileSet()

			// When: each source is parsed and its shard declarations
			// collected, keyed by file and variable name.
			shards := make(map[string]string)
			var parseErrors []error
			for _, source := range sources {
				file, err := parser.ParseFile(files, source, nil, parser.ParseComments)
				if err != nil {
					parseErrors = append(parseErrors, err)
					continue
				}
				for name, typ := range embeddedShardTypes(file) {
					shards[filepath.Base(source)+": "+name] = typ
				}
			}

			// Then
			for _, err := range parseErrors {
				t.Errorf("parse: %v", err)
			}
			if len(shards) == 0 {
				t.Fatalf("%v embed no %s files", sources, shardSuffix)
			}
			for name, typ := range shards {
				if typ != "[]byte" {
					t.Errorf("%s declares a shard as %s, want []byte", name, typ)
				}
			}
		})
	}
}

// shardModuleSources returns the non-test Go source files of every module
// that embeds the named real model's shards, failing t when it cannot list
// them or finds none.
func shardModuleSources(t *testing.T, model string) []string {
	t.Helper()
	var sources []string
	for _, dir := range shardModuleDirs(t, model) {
		matches, err := filepath.Glob(filepath.Join(dir, "*.go"))
		if err != nil {
			t.Fatalf("list Go files in %s: %v", dir, err)
		}
		for _, match := range matches {
			if !strings.HasSuffix(match, "_test.go") {
				sources = append(sources, match)
			}
		}
	}
	if len(sources) == 0 {
		t.Fatalf("no Go source files in the %s model's shard modules", model)
	}
	return sources
}

// embeddedShardTypes returns, for each variable in file whose //go:embed
// directive names a safetensors shard, the source text of its declared type.
func embeddedShardTypes(file *ast.File) map[string]string {
	shards := make(map[string]string)
	for _, declaration := range file.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.VAR {
			continue
		}
		for _, spec := range general.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			if !embedsShard(value.Doc) && !(len(general.Specs) == 1 && embedsShard(general.Doc)) {
				continue
			}
			for _, name := range value.Names {
				shards[name.Name] = typeText(value.Type)
			}
		}
	}
	return shards
}

// embedsShard reports whether doc holds a //go:embed directive naming a
// safetensors file. doc may be nil.
func embedsShard(doc *ast.CommentGroup) bool {
	if doc == nil {
		return false
	}
	for _, comment := range doc.List {
		if strings.HasPrefix(comment.Text, embedDirective) && strings.HasSuffix(strings.TrimSpace(comment.Text), shardSuffix) {
			return true
		}
	}
	return false
}

// typeText returns the source form of the simple type expressions a shard
// declaration can use, such as "string" or "[]byte"; any other form is
// returned as "unsupported".
func typeText(expression ast.Expr) string {
	switch typed := expression.(type) {
	case *ast.Ident:
		return typed.Name
	case *ast.ArrayType:
		if typed.Len == nil {
			return "[]" + typeText(typed.Elt)
		}
	}
	return "unsupported"
}
