package parity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// upstreamCheckpoint identifies the published checkpoint a model's files
// came from.
type upstreamCheckpoint struct {
	// Repository is the Hugging Face repository.
	Repository string `json:"repository"`
	// Revision is the pinned commit of that repository.
	Revision string `json:"revision"`
	// SHA256 is the digest of the upstream model.safetensors.
	SHA256 string `json:"sha256"`
}

// manifestShard describes one shard a manifest lists.
type manifestShard struct {
	// File is the shard's file name within weights/.
	File string `json:"file"`
	// Bytes is the shard's size.
	Bytes int64 `json:"bytes"`
	// SHA256 is the shard's digest.
	SHA256 string `json:"sha256"`
}

// weightsManifest mirrors the weights/manifest.json of a model module or of a
// module that embeds a model's shards. A model module's manifest carries
// ConfigSHA256; one that embeds shards carries MaxShardBytes and Shards; the
// small model's module carries all three. The base model's module also lists
// the modules that embed its shards in ShardModules.
type weightsManifest struct {
	// Upstream identifies the published checkpoint the files came from.
	Upstream upstreamCheckpoint `json:"upstream"`
	// MaxShardBytes is the shard-size limit the shards were cut to.
	MaxShardBytes int64 `json:"max_shard_bytes"`
	// ConfigSHA256 is the digest of weights/config.json.
	ConfigSHA256 string `json:"config_sha256"`
	// ShardModules lists, in checkpoint order, the paths of the modules that
	// embed the shards, when they are not in this module.
	ShardModules []string `json:"shard_modules"`
	// Shards lists every shard this module embeds, with its size and digest.
	Shards []manifestShard `json:"shards"`
}

// fileSHA256 returns the hex SHA-256 digest and size of the file at path.
func fileSHA256(t *testing.T, path string) (string, int64) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), int64(len(data))
}

// TestWeightsManifest_EachModel_MatchesCommittedShards verifies that each
// model's manifests match its committed config and shards, in its own module
// and in every module that embeds its shards.
func TestWeightsManifest_EachModel_MatchesCommittedShards(t *testing.T) {
	t.Parallel()

	models := []struct {
		name         string
		upstream     upstreamCheckpoint
		shardModules []string
	}{
		{"base", upstreamCheckpoint{"amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"}, shardModulePaths["base"]},
		{"small", upstreamCheckpoint{"autogluon/chronos-2-small", "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a", "492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd"}, nil},
	}
	for _, model := range models {
		t.Run(model.name, func(t *testing.T) {
			t.Parallel()

			// Given: the model module's manifest and the weights directories
			// of the modules that embed its shards, read from wherever the go
			// command resolves each module.
			weights := filepath.Join(modelModuleDir(t, model.name), "weights")
			manifest := readWeightsManifest(t, weights)
			shardWeights := shardWeightsDirs(t, model.name)

			// When: the config and every shard module's files are observed.
			configSum, _ := fileSHA256(t, filepath.Join(weights, "config.json"))
			observed := make([]shardModuleFiles, len(shardWeights))
			for i, dir := range shardWeights {
				observed[i] = observeShardModule(t, dir)
			}

			// Then: every manifest names the pinned upstream checkpoint and
			// matches the committed files.
			if manifest.Upstream != model.upstream {
				t.Errorf("%s upstream = %+v, want %+v", weights, manifest.Upstream, model.upstream)
			}
			if configSum != manifest.ConfigSHA256 {
				t.Errorf("config.json SHA-256 = %s, manifest says %s", configSum, manifest.ConfigSHA256)
			}
			if !slices.Equal(manifest.ShardModules, model.shardModules) {
				t.Errorf("shard_modules = %v, want %v", manifest.ShardModules, model.shardModules)
			}
			for _, files := range observed {
				checkShardModule(t, files, model.upstream)
			}
		})
	}
}

// shardModuleFiles is what one shard module's weights directory holds: its
// manifest, the observed digest of every shard that manifest lists, and the
// number of safetensors files on disk.
type shardModuleFiles struct {
	// dir is the weights directory.
	dir string

	// manifest is the directory's decoded manifest.json.
	manifest weightsManifest

	// digests holds the observed digest of each listed shard, in the
	// manifest's order.
	digests []shardDigest

	// onDisk is the number of safetensors files in the directory.
	onDisk int
}

// observeShardModule reads the manifest in the weights directory dir, hashes
// every shard it lists, and counts the safetensors files present, failing t
// when a file cannot be read.
func observeShardModule(t *testing.T, dir string) shardModuleFiles {
	t.Helper()
	manifest := readWeightsManifest(t, dir)
	entries, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		t.Fatalf("list shards in %s: %v", dir, err)
	}
	return shardModuleFiles{dir: dir, manifest: manifest, digests: hashShards(t, dir, manifest), onDisk: len(entries)}
}

// checkShardModule reports, through t, every way files departs from its
// manifest: an upstream other than want, a shard whose digest or size differs
// from the listing or exceeds the shard-size limit, and a shard on disk that
// the manifest does not list.
func checkShardModule(t *testing.T, files shardModuleFiles, want upstreamCheckpoint) {
	t.Helper()
	if files.manifest.Upstream != want {
		t.Errorf("%s upstream = %+v, want %+v", files.dir, files.manifest.Upstream, want)
	}
	if len(files.manifest.Shards) == 0 {
		t.Errorf("%s: the manifest lists no shards", files.dir)
	}
	for i, shard := range files.manifest.Shards {
		if files.digests[i].sum != shard.SHA256 || files.digests[i].size != shard.Bytes {
			t.Errorf("%s: SHA-256 %s, %d bytes; manifest says %s, %d bytes", shard.File, files.digests[i].sum, files.digests[i].size, shard.SHA256, shard.Bytes)
		}
		if shard.Bytes > files.manifest.MaxShardBytes {
			t.Errorf("%s: %d bytes exceeds the %d-byte shard limit", shard.File, shard.Bytes, files.manifest.MaxShardBytes)
		}
	}
	if files.onDisk != len(files.manifest.Shards) {
		t.Errorf("%s holds %d shards; the manifest lists %d", files.dir, files.onDisk, len(files.manifest.Shards))
	}
}

// shardDigest is the observed SHA-256 sum and size of one shard file.
type shardDigest struct {
	// sum is the file's hex-encoded SHA-256 sum.
	sum string

	// size is the file's length in bytes.
	size int64
}

// readWeightsManifest decodes weights/manifest.json from the weights
// directory, failing t if it cannot.
func readWeightsManifest(t *testing.T, weights string) weightsManifest {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(weights, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest weightsManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	return manifest
}

// hashShards returns the digest of every shard the manifest lists, in its
// order.
func hashShards(t *testing.T, weights string, manifest weightsManifest) []shardDigest {
	t.Helper()
	shards := make([]shardDigest, len(manifest.Shards))
	for i, shard := range manifest.Shards {
		shards[i].sum, shards[i].size = fileSHA256(t, filepath.Join(weights, shard.File))
	}
	return shards
}
