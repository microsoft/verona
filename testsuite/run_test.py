#!/usr/bin/env python3

# Verona Test Runner
#
# Syntax:
#   $ run_test.py path/to/test
#
# Assumptions:
#   * Test has at least one RUN line in the format:
#     // RUN: program [args] %s [args]
#   * '%s' is substituted by the path+file
#   * The golden-file is at: path/golden/file.out
#
# Functionality:
#
# First, we create a directory at path/output (if not exists yet). This is
# where all the command outputs will be stored at.
#
# The idea is to read the test file, and create a RUN job for each RUN line.
# The RUN job will substitute '%s' with the file name and run that as a command,
# redirecting the output to a file at path/output/file.out.N. Then it will
# compare that output with the golden file (at path/golden.out) and mark as
# error if the files differ (showing the difference), or mark as success if not.

import argparse
import sys
import os
import re
import subprocess
import difflib

"""
    TestFile

    Simple wrapper to a test file, collects all RUN lines and prepares both
    output and golden locations for the TestRunner.
"""
class TestFile:
    def __init__(self, name):
        self.name = os.path.realpath(name)
        self.golden = os.path.realpath(os.path.join(os.path.join(os.path.join(
                        self.name, os.pardir), 'golden'), name+'.out'))
        self.output = os.path.realpath(os.path.join(os.path.join(os.path.join(
                        self.name, os.pardir), 'output'), name+'.out'))

    """Return a list of RUN lines, split like a command line and with %s
       substituted with the file name"""
    def runners(self):
        lines = list()
        pattern = re.compile(r'[\/#]+ RUN: (.*)')
        space = re.compile(r' +')
        repl = re.compile(r'%s')

        with open(self.name, 'r') as f:
            # For each RUN line
            for line in f:
                match = pattern.match(line)
                if not match:
                    continue
                cmd = match.group(1)
                # Replace %s with file name
                cmd = repl.sub(self.name, cmd);
                # Append
                lines.append(space.split(cmd))

        return lines

"""
    TestRunner

    Runs the test and diffs the output for error reporting.
"""
class TestRunner:
    def __init__(self, args, golden, output, index):
        self.args = args
        self.golden = golden + '.' + repr(index)
        self.output = output + '.' + repr(index)
        self.index = index
        output_dir = os.path.realpath(os.path.dirname(self.output))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.status = 0

    """Run the command line, returning the exit status"""
    def run(self):
        with open(self.output, 'w') as out:
            process = subprocess.Popen(self.args, stdout=out, stderr=out)
            self.status = process.wait()
            return self.status

    """Compare the output with the golden file,
       returning the difference if any"""
    def diff(self):
        if not os.path.exists(self.golden):
            print("Invalid path to golden file for comparison", self.golden)
            return
        if not os.path.exists(self.output):
            print("Invalid path to output file for comparison", self.output)
            return

        with open(self.output, 'r') as out, open(self.golden, 'r') as gold:
            left = gold.readlines()
            right = out.readlines()
            if left != right:
                d = difflib.Differ()
                return list(d.compare(left, right))

        return None

if __name__ == "__main__":
    # There is only one argument, the test file
    parser = argparse.ArgumentParser()
    parser.add_argument('test',
        help='Test file'
    )
    args = parser.parse_args()

    # Creates the test file, with output and golden paths
    test = TestFile(args.test);
    index = 1
    status = 0

    # For each RUN line
    for run in test.runners():
        # Create a runner with file, output and golden locations
        runner = TestRunner(run, test.golden, test.output, index)

        # Run the test, if errors (return is non-zero on error)
        if runner.run():
            print(test.name, "test", index, "FAIL", runner.args)
            with open(runner.output, 'r') as out:
                sys.stdout.writelines(out)
            status = -1
            continue

        # Show the diff, if any
        diff = runner.diff()
        if diff is None:
            print(test.name, "test", index, "PASS")
        else:
            print(test.name, "test", index, "FAIL", runner.args)
            sys.stdout.writelines(diff)
        index += 1

    # Return non-zero on error
    sys.exit(status)
