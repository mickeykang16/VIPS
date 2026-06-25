#!/usr/bin/env bash

# Download and restore generated V2X-Real evaluation assets from HuggingFace.
# This does not download the official V2X-Real raw test images; place those at
# $V2XREAL_ROOT/data/test from the official UCLA release.

set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-mickeykang/VIPS-v2xreal-assets}"
: "${V2XREAL_ROOT:?Set V2XREAL_ROOT to the final V2X-Real evaluation data root.}"
HF_ASSET_DIR="${HF_ASSET_DIR:-$V2XREAL_ROOT/_hf_assets}"

HF_CLI="${HF_CLI:-}"
if [[ -z "$HF_CLI" ]]; then
    if command -v hf >/dev/null 2>&1; then
        HF_CLI="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI="huggingface-cli"
    fi
fi

if [[ -z "$HF_CLI" ]]; then
    echo "ERROR: HuggingFace CLI not found. Install it with: pip install -U huggingface_hub"
    exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
    echo "ERROR: tar not found."
    exit 1
fi

mkdir -p "$HF_ASSET_DIR" "$V2XREAL_ROOT/data" "$V2XREAL_ROOT/infos/v2xreal/cooperative"

echo "Downloading HF assets"
echo "Repo:      $HF_REPO_ID"
echo "Local dir: $HF_ASSET_DIR"
if ! "$HF_CLI" download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$HF_ASSET_DIR"; then
    echo
    echo "ERROR: failed to download HF assets."
    echo "The repo is public; if it is ever gated/private, run 'hf auth login' (or 'huggingface-cli login') with an account that has access."
    exit 1
fi

if [[ -f "$HF_ASSET_DIR/archives/SHA256SUMS" ]] && command -v sha256sum >/dev/null 2>&1; then
    echo "Checking archive SHA256 sums"
    (cd "$HF_ASSET_DIR/archives" && sha256sum -c SHA256SUMS)
fi

if [[ -f "$HF_ASSET_DIR/meta/maps_final/v2x_real_map.json" ]]; then
    mkdir -p "$V2XREAL_ROOT/maps_final"
    cp -a "$HF_ASSET_DIR/meta/maps_final/v2x_real_map.json" "$V2XREAL_ROOT/maps_final/"
else
    echo "ERROR: missing $HF_ASSET_DIR/meta/maps_final/v2x_real_map.json"
    exit 1
fi

if [[ -f "$HF_ASSET_DIR/meta/spd_infos_temporal_test.pkl" ]]; then
    cp -a "$HF_ASSET_DIR/meta/spd_infos_temporal_test.pkl" \
        "$V2XREAL_ROOT/infos/v2xreal/cooperative/"
else
    echo "ERROR: missing $HF_ASSET_DIR/meta/spd_infos_temporal_test.pkl"
    exit 1
fi

extract_archive_parts() {
    local name="$1"
    local archive_dir="$HF_ASSET_DIR/archives"
    local direct_dir=""
    local -a parts=()

    mapfile -t parts < <(find "$archive_dir" -maxdepth 1 -type f -name "${name}.tar.zst.part-*" -print 2>/dev/null | sort)
    if (( ${#parts[@]} > 0 )); then
        echo "Extracting ${name} from ${#parts[@]} archive shards"
        cat "${parts[@]}" | tar --zstd -xf - -C "$V2XREAL_ROOT/data"
        return
    fi

    if [[ -d "$HF_ASSET_DIR/data/$name" ]]; then
        direct_dir="$HF_ASSET_DIR/data/$name"
    elif [[ -d "$HF_ASSET_DIR/$name" ]]; then
        direct_dir="$HF_ASSET_DIR/$name"
    fi

    if [[ -n "$direct_dir" ]]; then
        echo "Copying ${name} directory"
        cp -a "$direct_dir" "$V2XREAL_ROOT/data/"
        return
    fi

    echo "ERROR: missing ${name} archive shards or directory in $HF_ASSET_DIR"
    exit 1
}

extract_archive_parts "test_novel"
extract_archive_parts "test_novel_infra"

echo
echo "Generated assets restored under: $V2XREAL_ROOT"
if [[ ! -d "$V2XREAL_ROOT/data/test" ]]; then
    echo "WARNING: $V2XREAL_ROOT/data/test is missing."
    echo "Download the official V2X-Real-Lidar-Cameras test set from UCLA and place it there."
fi
