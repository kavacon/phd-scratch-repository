# Source this to put every folder's Python package on the path: `. ./env.sh` from this directory.
here="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
PYTHONPATH=$(ls -d "$here"/*/python | tr '\n' ':')
export PYTHONPATH
