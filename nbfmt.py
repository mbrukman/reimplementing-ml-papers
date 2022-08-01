#!/usr/bin/python
#
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonicalizes field ordering and values in Jupyter notebooks.

This enables minimal human-reviewable diffs from one version to the next, even
when updated in various tools like Colab, VS Code, etc., each of which make
their own choices in various regards like field ordering, add or update
inconsequential fields and their values, leading to spurious diffs that obscure
the real changes.
"""

import json
import sys
from typing import Dict, List


EXECUTION_COUNT = 'execution_count'


def processList(data: List):
    """Processes the passed-in list recursively, may modify it in-place."""
    for item in data:
        if isinstance(item, list):
            processList(item)
        elif isinstance(item, dict):
            processDict(item)


def processDict(data: Dict):
    """Processes the passed-in dict recursively, may modify it in-place."""
    # Reset execution counts for code cells.
    if EXECUTION_COUNT in data:
        data[EXECUTION_COUNT] = None

    for key in data.keys():
        if isinstance(data[key], list):
            processList(data[key])
        elif isinstance(data[key], dict):
            processDict(data[key])


# TODO(mbrukman): add flag `-w` to rewrite the file in-place, a la gofmt.
def main(argv):
    """Parses notebook and outputs canonicalized version to stdout."""
    if len(argv) < 2:
        sys.stderr.write(f'Syntax: {argv[0]} [path-to-notebook]\n')
        sys.exit(1)

    notebook = argv[1]
    json_input = None
    with open(notebook, 'r') as json_file:
        json_input = json.loads(json_file.read())

    # Updates the JSON in-place.
    processDict(json_input)

    # Apply a few more canonicalization rules:
    #
    # * avoid newline at the end of file;
    # * `sort_keys` fixes field ordering inside JSON objects;
    # * use 2-space indent.
    sys.stdout.write(json.dumps(json_input, sort_keys=True, indent=2))


if __name__ == '__main__':
    main(sys.argv)
