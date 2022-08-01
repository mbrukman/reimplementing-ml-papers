#!/bin/bash -u
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

function clean_file() {
  local file="$1"
  local temp="${file}.tmp"

  echo -n "Updating ${file} ... "
  python "$(dirname $0)/nbfmt.py" "${file}" > "${temp}"
  if diff "${file}" "${temp}" > /dev/null 2>&1; then
    rm "${temp}"
    echo "no change."
  else
    mv "${temp}" "${file}"
    echo "done."
  fi
}

for file in $(find -s . -name \*\.ipynb); do
  clean_file "${file}"
done
