#!/usr/bin/env bash
# Regenerate the deliberately minimal Claude/Codex host entrypoints.
#
# Canonical internal modules remain under .agent/skills. The generator exposes
# only 00-router and explicit grill-me, then removes stale wrappers when clean.
#
# Usage:
#   bash .agent/scripts/sync_agent_skill_wrappers.sh [REPO_ROOT] [both|claude|codex] [--clean]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REPO_ROOT="${1:-$DEFAULT_REPO_ROOT}"
TARGET="${2:-both}"
CLEAN_FLAG=""
if [[ "${3:-}" == "--clean" || "${CLEAN_ENV:-0}" == "1" ]]; then
  CLEAN_FLAG="--clean"
fi

# Refuse linked destinations before either host is changed. The generator writes
# through existing paths, so leaving a symlink in place could overwrite its target.
for HOST in claude codex; do
  if [[ "$TARGET" != "both" && "$TARGET" != "$HOST" ]]; then
    continue
  fi
  DESTINATION="$REPO_ROOT/.$HOST/skills"
  ANCESTOR="$DESTINATION"
  while [[ "$ANCESTOR" != "/" && "$ANCESTOR" != "." ]]; do
    if [[ -L "$ANCESTOR" ]]; then
      echo "ERROR: refusing symlink destination: $ANCESTOR" >&2
      exit 1
    fi
    ANCESTOR="$(dirname "$ANCESTOR")"
  done
  if [[ -d "$DESTINATION" ]]; then
    LINK="$(find -P "$DESTINATION" -type l -print -quit)"
    if [[ -n "$LINK" ]]; then
      echo "ERROR: refusing symlink within host skills: $LINK" >&2
      exit 1
    fi
  fi
done


python "$SCRIPT_DIR/generate_agent_skill_wrappers.py" \
  --repo-root "$REPO_ROOT" \
  --source-dir ".agent/skills" \
  --target "$TARGET" \
  $CLEAN_FLAG

python "$SCRIPT_DIR/validate_agent_skill_wrappers.py" \
  --canonical "$REPO_ROOT/.agent/skills" \
  --claude "$REPO_ROOT/.claude/skills" \
  --codex "$REPO_ROOT/.codex/skills" \
  --target "$TARGET"

echo
echo "Done. Canonical internal skills: $REPO_ROOT/.agent/skills"
echo "Host-visible allowlist: 00-router, grill-me"
if [ "$TARGET" = "both" ] || [ "$TARGET" = "claude" ]; then
  echo "  Claude entrypoints: $REPO_ROOT/.claude/skills"
fi
if [ "$TARGET" = "both" ] || [ "$TARGET" = "codex" ]; then
  echo "  Codex entrypoints:  $REPO_ROOT/.codex/skills"
fi
echo "Restart Claude Code / Codex to pick up wrapper changes."
