#!/bin/bash
set -e -x

REPORTDIR="report"

function clean {
  if [ -d "$REPORTDIR" ]; then
    rm -r "$REPORTDIR"
  fi
  mkdir -p "$REPORTDIR"
}

clean
xz -d -k -f infra/points.json.xz
racket -y infra/time.rkt --dir "$REPORTDIR" --profile profile.json infra/points.json
cp profile.json "$REPORTDIR"/profile.json
cp infra/profile.js "$REPORTDIR"/profile.js
