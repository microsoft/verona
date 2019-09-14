#!/usr/bin/env python2

# Compare the dump output of the Verona compiler to a set of expected files.
#
# Every .txt file in the expected set must exist in the dump output. The dump is
# allowed to have additional files which will be ignored.

from __future__ import print_function

import os
import os.path
import sys
import fnmatch
import difflib

if len(sys.argv) != 3:
  print("Usage: %s EXPECTED ACTUAL" % sys.argv[0], file=sys.stderr)
  sys.exit(1)

EXPECTED_DIR = sys.argv[1]
ACTUAL_DIR = sys.argv[2]

ERRORS = []
DIFF = []

def check_file(filename):
  actual_path = os.path.join(ACTUAL_DIR, filename)
  expected_path = os.path.join(EXPECTED_DIR, filename)

  if not os.path.exists(actual_path):
    ERRORS.append("Dump file %r is missing\n" % filename)
    return

  with open(actual_path, "rU") as actual, open(expected_path, "rU") as expected:
    diff = list(difflib.unified_diff(
      expected.readlines(), actual.readlines(),
      fromfile=os.path.join('expected', filename),
      tofile=os.path.join('actual', filename)))

    if diff:
      ERRORS.append("Found differences in %r\n" % filename)
      DIFF.append('\n')
      DIFF.extend(diff)

for name in fnmatch.filter(os.listdir(EXPECTED_DIR), '*.txt'):
  check_file(name)

sys.stderr.writelines(ERRORS)
sys.stderr.writelines(DIFF)

if DIFF:
  print("\nRun 'ninja update-dump' to update the testsuite.", file=sys.stderr)

if ERRORS:
  sys.exit(1)
