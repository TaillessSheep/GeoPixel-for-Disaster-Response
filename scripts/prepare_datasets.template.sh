#!/bin/bash

set -u

API_KEY="GOOGLE-DRIVE_API-KEY"

CONCURRENCY_LIMIT=20
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_ROOT="$SCRIPT_DIR/../data"
FINETUNE_DIR="$DATA_ROOT/GeoPixelD/finetune"
DATA_TXT="$SCRIPT_DIR/../data.txt"

ID_PNG_FOLDER="1EXM9g6uvPGw_SPdEo83scnR9dtHi6fIT"
ID_JSON_FOLDER="1Uq7ruYhaxdVfq4LKsAExJdjIP0397nqY"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

download_worker() {
    local fid="$1"
    local filename="$2"
    local target_dir="$3"
    local cookie_file="/tmp/cookie_${fid}.txt"
    local filepath="$target_dir/$filename"

    if [[ -s "$filepath" ]]; then
        return
    fi

    local url="https://drive.google.com/uc?export=download&id=${fid}"

    local confirm_code
    confirm_code=$(curl -s -c "$cookie_file" -L "$url" | grep -o 'confirm=[a-zA-Z0-9]*' | grep -o '[a-zA-Z0-9]*$' | head -n1)

    if [ -n "$confirm_code" ]; then
        curl -s -L -b "$cookie_file" -o "$filepath" "${url}&confirm=${confirm_code}" --retry 3
    else
        curl -s -L -c "$cookie_file" -o "$filepath" "$url" --retry 3
    fi

    if [ $? -eq 0 ] && [ -s "$filepath" ]; then
        echo -e "${GREEN}[DOWN]${NC} Done: $filename"
    else
        echo -e "${RED}[FAIL]${NC} Failed: $filename"
        rm -f "$filepath"
    fi
    rm -f "$cookie_file"
}

fetch_file_list() {
    local folder_id="$1"
    local page_token=""
    local url

    local list_file="/tmp/gdrive_list_${folder_id}.txt"
    > "$list_file"

    log_info "Fetching file list via API for folder: $folder_id"

    while :; do
        url="https://www.googleapis.com/drive/v3/files?q='${folder_id}'+in+parents+and+trashed=false&key=${API_KEY}&fields=nextPageToken,files(id,name)&pageSize=1000"

        if [ -n "$page_token" ]; then
            url="${url}&pageToken=${page_token}"
        fi

        local response
        response=$(curl -s -L "$url")

        if echo "$response" | grep -q "error"; then
            log_warn "API Error. Please check your API Key."
            echo "$response"
            return 1
        fi

        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for f in data.get('files', []):
        name = f['name'].replace('\n', '').replace('\r', '')
        fid = f['id']
        print(fid + '\t' + name)
except Exception as e:
    sys.stderr.write(str(e) + '\n')
" >> "$list_file"

        page_token=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('nextPageToken', ''))")

        if [ -z "$page_token" ]; then
            break
        fi
    done
    echo ""
}

process_folder_parallel() {
    local folder_id="$1"
    local target_dir="$2"
    local ext_filter="$3"

    fetch_file_list "$folder_id"
    local list_file="/tmp/gdrive_list_${folder_id}.txt"

    if [ ! -s "$list_file" ]; then
        log_warn "No files found or API failed for $folder_id"
        return
    fi

    local total_files
    total_files=$(wc -l < "$list_file")
    log_info "Found $total_files files via API. Starting download..."

    local job_count=0

    while IFS=$'\t' read -r fid fname; do
        if [[ "$fname" != *."$ext_filter" ]]; then
            continue
        fi

        download_worker "$fid" "$fname" "$target_dir" &

        ((job_count++))
        if [[ "$job_count" -ge "$CONCURRENCY_LIMIT" ]]; then
            wait
            job_count=0
        fi

    done < "$list_file"

    wait
    rm -f "$list_file"
    log_ok "Folder processing complete: $target_dir"
}

main() {
    if [[ "$API_KEY" == "YOUR_GOOGLE_API_KEY_HERE" ]] || [[ -z "$API_KEY" ]]; then
        echo -e "${RED}[ERROR]${NC} You must set your Google API Key in the script!"
        echo "Please get a free key from Google Cloud Console."
        exit 1
    fi

    log_info "=== Stable Download Script Started (Clean & Sync Mode) ==="


    if [ -d "$DATA_ROOT" ]; then
        log_warn "Wiping existing data directory: $DATA_ROOT"
        rm -rf "$DATA_ROOT"
    fi

    log_info "Recreating directory structure..."
    mkdir -p "$FINETUNE_DIR"
    rm -f "$DATA_TXT"
    touch "$DATA_TXT"

    process_folder_parallel "$ID_PNG_FOLDER" "$FINETUNE_DIR" "png"

    process_folder_parallel "$ID_JSON_FOLDER" "$DATA_ROOT" "json"

    log_info "Generating data.txt..."

    local added=0
    shopt -s nullglob
    for json_path in "$DATA_ROOT"/*.json; do
        local fname
        fname=$(basename "$json_path")
            echo "data/$fname 1.00" >> "$DATA_TXT"
            ((added++))
    done
    shopt -u nullglob

    log_ok "All done! data.txt entries: $added"
}

main