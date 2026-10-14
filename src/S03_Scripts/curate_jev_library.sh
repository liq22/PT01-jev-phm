#!/usr/bin/env bash
# Reproduce the authorized-source download and ZIP-skill archive for this list.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
records=paper/library/jev-generalist-20/references.json
stage=.venv/paper-curator
skill="$stage/skill/project-paper-curator"
python=${CURATOR_PYTHON:-$stage/runtime/bin/python}
zip=${CURATOR_ZIP:-.venv/project-paper-curator.zip}

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --download ) ]]; then
    printf 'Usage: %s [--download]\n' "$0" >&2
    exit 2
fi
command -v jq >/dev/null
[[ -x "$python" && -f "$zip" ]] || {
    printf 'Supply the curator ZIP and isolated Python environment; see the library README.\n' >&2
    exit 2
}
mkdir -p "$stage/skill" "$stage/downloads" "$stage/checks"
unzip -n "$zip" 'project-paper-curator/*' -d "$stage/skill" >/dev/null

if [[ ${1:-} == --download ]]; then
    while IFS=$'\t' read -r rid url; do
        [[ -f "$stage/downloads/$rid.pdf" ]] && continue
        if curl --fail --silent --show-error --location \
            --connect-timeout 15 --max-time 90 --retry 2 --retry-delay 6 \
            "$url" -o "$stage/downloads/$rid.part"; then
            mv -- "$stage/downloads/$rid.part" "$stage/downloads/$rid.pdf"
        else
            printf '%s: source download failed; retain a missing entry.\n' "$rid" >&2
        fi
        sleep 6
    done < <(jq -r '.references[] | select(.download_url != null and .download_url != "") | [.rid, .download_url] | @tsv' "$records")
fi

# The public records describe the observed batch. Construct inputs from files
# actually present here, without carrying their old nested verification report.
while IFS= read -r row; do
    rid=$(jq -r '.rid' <<<"$row")
    if [[ -f "$stage/downloads/$rid.pdf" ]]; then
        jq -c --arg path "downloads/$rid.pdf" \
            'del(.verification) | .local_pdf=$path | .available_locally=true' <<<"$row"
    else
        jq -c 'del(.verification, .local_pdf) | .available_locally=false |
            .download_status="missing" |
            if (.reason == null or .reason == "") then
                .reason="Local source PDF not supplied; see its recorded acquisition note and stable source."
            else . end' <<<"$row"
    fi
done < <(jq -c '.references[]' "$records") > "$stage/local-records.jsonl"
jq -s '{batch:"jev-generalist-20-20261014",references:.}' \
    "$stage/local-records.jsonl" > "$stage/local-records.json"

while IFS= read -r row; do
    rid=$(jq -r '.rid' <<<"$row")
    "$python" "$skill/scripts/verify_pdf.py" \
        --pdf "$stage/$(jq -r '.local_pdf' <<<"$row")" \
        --title "$(jq -r '.title' <<<"$row")" \
        --authors "$(jq -r '.author' <<<"$row")" \
        --doi "$(jq -r '.doi // ""' <<<"$row")" \
        --out "$stage/checks/$rid.json"
    expected=$(jq -r --arg rid "$rid" \
        '.references[] | select(.rid==$rid) | .verification.sha256 // ""' "$records")
    actual=$(jq -r '.sha256' "$stage/checks/$rid.json")
    if [[ -z "$expected" || "$actual" != "$expected" ]]; then
        printf '%s: input differs from the reviewed PDF; review its exact version and license and update references.json before archiving.\n' "$rid" >&2
        exit 1
    fi
done < <(jq -c '.references[] | select(.local_pdf != null)' "$stage/local-records.json")

# The complete local archive includes restricted PDFs. The tracked archive is
# a permission-checked subset with the same UIDs and original project numbers.
jq '.references |= map(select(.public_redistribution_allowed == true and .local_pdf != null))' \
    "$stage/local-records.json" > "$stage/public-records.json"
"$python" "$skill/scripts/archive_papers.py" \
    --records "$stage/local-records.json" --output "$stage/library" --dry-run
"$python" "$skill/scripts/archive_papers.py" \
    --records "$stage/local-records.json" --output "$stage/library"
"$python" "$skill/scripts/archive_papers.py" \
    --records "$stage/public-records.json" --output paper/library --dry-run
"$python" "$skill/scripts/archive_papers.py" \
    --records "$stage/public-records.json" --output paper/library
