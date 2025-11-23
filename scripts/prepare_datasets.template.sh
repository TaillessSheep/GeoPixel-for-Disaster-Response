#!/bin/bash

set -u

CONCURRENCY_LIMIT=20
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_ROOT="$SCRIPT_DIR/../data"
FINETUNE_DIR="$DATA_ROOT/GeoPixelD/finetune"
DATA_TXT="$SCRIPT_DIR/../data.txt"

ID_PNG_FOLDER="1EXM9g6uvPGw_SPdEo83scnR9dtHi6fIT"
ID_JSON_FOLDER="1l4E01xpRqP9mXiZxzTTySSVOKGQH-lP_"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }

download_worker() {
    local fid="$1"
    local target_dir="$2"
    local ext="$3"
    local cookie_file="/tmp/cookie_${fid}.txt"

    local header_info
    header_info=$(curl -s -I -L --retry 3 "https://drive.google.com/uc?export=download&id=$fid")

    local filename
    filename=$(echo "$header_info" | sed -n 's/.*filename="\([^"]*\)".*/\1/p')

    if [[ -z "$filename" ]]; then
        filename=$(echo "$header_info" | sed -n "s/.*filename\*=UTF-8''\([^[:space:]]*\).*/\1/p")
        filename=$(echo "$filename" | sed 's/%20/ /g' | sed 's/%2E/./g')
    fi
    filename=$(echo "$filename" | tr -d '\r\n')

    if [[ "$filename" == *."$ext" ]]; then
        local url="https://drive.google.com/uc?export=download&id=${fid}"
        local confirm_code
        confirm_code=$(curl -s -c "$cookie_file" -L "$url" | grep -o 'confirm=[a-zA-Z0-9]*' | grep -o '[a-zA-Z0-9]*$' | head -n1)

        if [ -n "$confirm_code" ]; then
            curl -s -L -b "$cookie_file" -o "$target_dir/$filename" "${url}&confirm=${confirm_code}" --retry 3
        else
            curl -s -L -c "$cookie_file" -o "$target_dir/$filename" "$url" --retry 3
        fi

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[DOWN]${NC} Done: $filename"
        else
            echo -e "${RED}[FAIL]${NC} $filename"
        fi
        rm -f "$cookie_file"
    fi
}

process_folder_parallel() {
    local folder_id="$1"
    local target_dir="$2"
    local ext="$3"

    log_info "Parsing file list (ID: $folder_id)..."

    local html_file="/tmp/gdrive_${folder_id}.html"
    curl -s -L "https://drive.google.com/drive/folders/${folder_id}" -A "Mozilla/5.0" > "$html_file"

    local id_list
    id_list=$(grep -oE '[a-zA-Z0-9_-]{33}' "$html_file" | sort -u)
    rm -f "$html_file"

    log_info "Starting concurrent download (Concurrency: $CONCURRENCY_LIMIT)..."

    local job_count=0

    for fid in $id_list; do
        if [[ "$fid" == "$folder_id" ]]; then continue; fi

        download_worker "$fid" "$target_dir" "$ext" &

        ((job_count++))

        if [[ "$job_count" -ge "$CONCURRENCY_LIMIT" ]]; then
            wait
            job_count=0
        fi
    done

    wait
    log_ok "Folder processing complete: $target_dir"
}

main() {
    log_info "=== Fast Download Script Started ==="

    log_info "Cleaning up old data..."

    if [ -d "$FINETUNE_DIR" ]; then
        rm -rf "$FINETUNE_DIR"
    fi
    mkdir -p "$FINETUNE_DIR"

    if [ -f "$DATA_TXT" ]; then
        rm -f "$DATA_TXT"
    fi

    if [ -d "$DATA_ROOT" ]; then
        log_info "Removing old JSON files in $DATA_ROOT..."
        rm -f "$DATA_ROOT"/*.json
    fi
    mkdir -p "$DATA_ROOT"

    process_folder_parallel "$ID_PNG_FOLDER" "$FINETUNE_DIR" "png"

    process_folder_parallel "$ID_JSON_FOLDER" "$DATA_ROOT" "json"

    log_info "Generating data.txt..."
    touch "$DATA_TXT"

    local added=0
    shopt -s nullglob
    for json_path in "$DATA_ROOT"/*.json; do
        local fname
        fname=$(basename "$json_path")
        local entry="data/$fname 0.01"
        echo "$entry" >> "$DATA_TXT"
        ((added++))
    done
    shopt -u nullglob

    log_ok "All done! data.txt entries: $added"
}

main