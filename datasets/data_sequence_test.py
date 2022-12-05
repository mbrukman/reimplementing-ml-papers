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

from data_sequence import DataSequence, DataSequenceWithShuffling

import unittest


class DataSequenceTest(unittest.TestCase):

    def testEqualSizedBatches(self):
        data = DataSequence(num_items=75, batch_size=25)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], (0, 25))
        self.assertEqual(data[1], (25, 50))
        self.assertEqual(data[2], (50, 75))

    def testUnequalBatches(self):
        data = DataSequence(num_items=100, batch_size=32)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], (0, 32))
        self.assertEqual(data[1], (32, 64))
        self.assertEqual(data[2], (64, 96))
        self.assertEqual(data[3], (96, 100))

    def testBatchSizeLargerThanNumberOfItems(self):
        data = DataSequence(num_items=7, batch_size=16)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], (0, 7))


class DataSequenceWithShuffleTest(unittest.TestCase):

    def testWithShufflingEqualBatches(self):
        num_items = 75
        batch_size = 25

        data = DataSequenceWithShuffling(
            num_items=num_items, batch_size=batch_size, shuffle=True)
        self.assertEqual(len(data), 3)

        expected_data = [
          ((0, 25), batch_size),
          ((25, 50), batch_size),
          ((50, 75), batch_size),
        ]

        all_indices_seen = []
        for index, (expected_range, expected_length) in enumerate(expected_data):
            actual_range, actual_indices = data[index]
            self.assertEqual(expected_range, actual_range)
            self.assertEqual(expected_length, len(actual_indices))
            all_indices_seen.extend(actual_indices)

        # Ensure that each position was represented exactly once.
        self.assertEqual(sorted(set(all_indices_seen)), list(range(0, num_items)))

    def testWithoutShufflingEqualBatches(self):
        num_items = 75
        batch_size = 25

        data = DataSequenceWithShuffling(
            num_items=num_items, batch_size=batch_size, shuffle=False)
        self.assertEqual(len(data), 3)

        expected_data = [
          ((0, 25), list(range(0, 25))),
          ((25, 50), list(range(25, 50))),
          ((50, 75), list(range(50, 75))),
        ]

        all_indices_seen = []
        for index, (expected_range, expected_indices) in enumerate(expected_data):
            actual_range, actual_indices = data[index]
            self.assertEqual(expected_range, actual_range)
            self.assertEqual(expected_indices, actual_indices)
            all_indices_seen.extend(actual_indices)

        # Ensure that each position was represented exactly once.
        self.assertEqual(sorted(set(all_indices_seen)), list(range(0, num_items)))

    def testWithShufflingUnequalBatches(self):
        num_items = 100
        batch_size = 32

        data = DataSequenceWithShuffling(
            num_items=num_items, batch_size=batch_size, shuffle=True)
        self.assertEqual(len(data), 4)

        expected_data = [
          ((0, 32), batch_size),
          ((32, 64), batch_size),
          ((64, 96), batch_size),
          ((96, 100), len(range(96, 100))),
        ]

        all_indices_seen = []
        for index, (expected_range, expected_size) in enumerate(expected_data):
            actual_range, actual_indexes = data[index]
            self.assertEqual(expected_range, actual_range)
            self.assertEqual(expected_size, len(actual_indexes))
            all_indices_seen.extend(actual_indexes)

        # Ensure that each position was represented exactly once.
        self.assertEqual(
            sorted(set(all_indices_seen)), list(range(0, num_items)))

    def testWithoutShufflingUnequalBatches(self):
        num_items = 100
        batch_size = 32

        data = DataSequenceWithShuffling(
            num_items=num_items, batch_size=batch_size, shuffle=False)
        self.assertEqual(len(data), 4)

        expected_data = [
          ((0, 32), list(range(0, 32))),
          ((32, 64), list(range(32, 64))),
          ((64, 96), list(range(64, 96))),
          ((96, 100), list(range(96, 100))),
        ]

        all_indices_seen = []
        for index, (expected_range, expected_indexes) in enumerate(expected_data):
            actual_range, actual_indexes = data[index]
            self.assertEqual(expected_range, actual_range)
            self.assertEqual(expected_indexes, actual_indexes)
            all_indices_seen.extend(actual_indexes)

        # Ensure that each position was represented exactly once.
        self.assertEqual(all_indices_seen, list(range(0, num_items)))


if __name__ == '__main__':
    unittest.main()

