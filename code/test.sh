#!/bin/sh
# Run every test of the repository's Python code. Each folder's tests run in their own process, since every folder
# has a package called `tests`. Needs the virtual environment in this directory (.venv) and extra arguments go to unittest.
cd "$(dirname "$0")"
. ./env.sh
status=0
for folder in helper demo experiment_*; do
  [ -d "$folder/tests" ] || continue
  .venv/bin/python -m unittest discover -s "$folder/tests" -t "$folder" "$@" || status=1
done
.venv/bin/python -m unittest discover -s tests -t . "$@" || status=1
exit $status
