package parity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// weightsManifest mirrors a model module's weights/manifest.json.
type weightsManifest struct {
	// Upstream identifies the published checkpoint the shards came from.
	Upstream struct {
		// Repository is the Hugging Face repository.
		Repository string `json:"repository"`
		// Revision is the pinned commit of that repository.
		Revision string `json:"revision"`
		// SHA256 is the digest of the upstream model.safetensors.
		SHA256 string `json:"sha256"`
	} `json:"upstream"`
	// MaxShardBytes is the shard-size limit the shards were cut to.
	MaxShardBytes int64 `json:"max_shard_bytes"`
	// ConfigSHA256 is the digest of weights/config.json.
	ConfigSHA256 string `json:"config_sha256"`
	// Shards lists every shard with its size and digest.
	Shards []struct {
		// File is the shard's file name within weights/.
		File string `json:"file"`
		// Bytes is the shard's size.
		Bytes int64 `json:"bytes"`
		// SHA256 is the shard's digest.
		SHA256 string `json:"sha256"`
	} `json:"shards"`
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

// TestWeightsManifest_EachModel_MatchesCommittedShards verifies that each model's weights manifest matches its committed shards.
func TestWeightsManifest_EachModel_MatchesCommittedShards(t *testing.T) {
	t.Parallel()

	models := []struct {
		name, repository, revision, upstreamSHA256 string
	}{
		{"base", "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"},
		{"small", "autogluon/chronos-2-small", "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a", "492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd"},
	}
	for _, model := range models {
		t.Run(model.name, func(t *testing.T) {
			t.Parallel()

			// Given: the model module's manifest, read from wherever the go
			// command resolves the module.
			weights := filepath.Join(modelModuleDir(t, model.name), "weights")
			manifest := readWeightsManifest(t, weights)

			// When: every listed file is hashed.
			configSum, _ := fileSHA256(t, filepath.Join(weights, "config.json"))
			shards := hashShards(t, weights, manifest)
			entries, err := filepath.Glob(filepath.Join(weights, "*.safetensors"))
			if err != nil {
				t.Fatalf("list shards: %v", err)
			}

			// Then: the manifest names the pinned upstream checkpoint and
			// matches every committed file, and no shard is unlisted.
			if manifest.Upstream.Repository != model.repository || manifest.Upstream.Revision != model.revision || manifest.Upstream.SHA256 != model.upstreamSHA256 {
				t.Errorf("upstream = %+v, want %s@%s with SHA-256 %s", manifest.Upstream, model.repository, model.revision, model.upstreamSHA256)
			}
			if configSum != manifest.ConfigSHA256 {
				t.Errorf("config.json SHA-256 = %s, manifest says %s", configSum, manifest.ConfigSHA256)
			}
			for i, shard := range manifest.Shards {
				if shards[i].sum != shard.SHA256 || shards[i].size != shard.Bytes {
					t.Errorf("%s: SHA-256 %s, %d bytes; manifest says %s, %d bytes", shard.File, shards[i].sum, shards[i].size, shard.SHA256, shard.Bytes)
				}
				if shard.Bytes > manifest.MaxShardBytes {
					t.Errorf("%s: %d bytes exceeds the %d-byte shard limit", shard.File, shard.Bytes, manifest.MaxShardBytes)
				}
			}
			if len(entries) != len(manifest.Shards) {
				t.Errorf("weights/ holds %d shards; the manifest lists %d", len(entries), len(manifest.Shards))
			}
		})
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
