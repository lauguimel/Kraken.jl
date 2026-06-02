#!/usr/bin/env bash
#
# lint-implication-map.sh — validate an LLM module-implication map.
#
# Spec: docs/spec/llm-implication-map-v1.md (KRK-LLM-MAP-001).
# Pure bash + grep, no jq / python / external deps.
#
# Exits 0 iff the file:
#   1. starts with a YAML frontmatter block delimited by `---`,
#   2. contains all six mandatory `## <title>` section headers,
#   3. has a non-empty body under each of the six sections.
#
# Usage: scripts/lint-implication-map.sh <path-to-map.md>
#
set -u

PASS_PREFIX="PASS"
FAIL_PREFIX="FAIL"

fail() {
    echo "${FAIL_PREFIX}: $1"
    exit 1
}

# --- argument ---------------------------------------------------------------
if [ "$#" -ne 1 ]; then
    echo "usage: $0 <path-to-implication-map.md>"
    exit 2
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    fail "file not found: $FILE"
fi

# --- 1. YAML frontmatter ----------------------------------------------------
# First non-empty line must be exactly `---`, and there must be a second `---`.
FIRST_LINE=$(grep -m1 -n '.' "$FILE" | cut -d: -f1)
if [ "$FIRST_LINE" != "1" ]; then
    fail "frontmatter must start on line 1 (no leading blank lines)"
fi

# Count fence lines that are exactly `---` (allow trailing whitespace).
FENCE_COUNT=$(grep -c -E '^---[[:space:]]*$' "$FILE")
if [ "$FENCE_COUNT" -lt 2 ]; then
    fail "missing YAML frontmatter block delimited by '---' (found $FENCE_COUNT fence line(s), need >= 2)"
fi

# The very first line of the file must be the opening fence.
HEAD_LINE=$(head -n 1 "$FILE")
case "$HEAD_LINE" in
    ---*) : ;;
    *) fail "first line must be the opening frontmatter fence '---'" ;;
esac

# --- 2 & 3. mandatory sections present and non-empty ------------------------
# Exact header strings (level-2). Order is not enforced by the linter.
SECTIONS="## Public surface
## Reads from
## Writes to
## Backend constraints
## Failure modes
## Touch order"

# Iterate sections line by line (no subshell loop variable leakage needed).
while IFS= read -r SECTION; do
    [ -z "$SECTION" ] && continue

    # Presence: exact full-line match of the header.
    if ! grep -q -F -x "$SECTION" "$FILE"; then
        fail "missing mandatory section header: '$SECTION'"
    fi

    # Non-empty body: read the file line by line; once inside the section
    # (header matched), collect content lines until the next `## ` header or
    # EOF, and require at least one line with visible (non-whitespace) content.
    INBLK=0
    HAS_BODY=0
    while IFS= read -r LINE || [ -n "$LINE" ]; do
        if [ "$INBLK" -eq 1 ]; then
            case "$LINE" in
                "## "*) INBLK=0 ;;
                *)
                    # any non-whitespace character counts as body content
                    if printf '%s' "$LINE" | grep -q -E '[^[:space:]]'; then
                        HAS_BODY=1
                        INBLK=0
                    fi
                    ;;
            esac
        fi
        if [ "$LINE" = "$SECTION" ]; then
            INBLK=1
        fi
    done < "$FILE"

    if [ "$HAS_BODY" -eq 0 ]; then
        fail "section '$SECTION' has an empty body"
    fi
done <<EOF
$SECTIONS
EOF

# --- success ----------------------------------------------------------------
echo "${PASS_PREFIX}: $FILE (frontmatter + 6 sections present and non-empty)"
exit 0
