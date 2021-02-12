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

    """Return a list of RUN lines, each a list of pipe commands,
       if more than one, split like a command line and with %s
       substituted with the file name"""
    def runners(self):
        lines = list()
        pattern = re.compile(r'[\/#]+ RUN: (.*)')
        pipe = re.compile(r' *\| *')
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
                # Split by pipe, then by space
                commands = list()
                for cmd in pipe.split(cmd):
                    commands.append(space.split(cmd))
                # Append
                lines.append(commands)

        return lines

"""
    TestRunner

    Runs the test and diffs the output for error reporting.
"""
class TestRunner:
    def __init__(self, args, golden, output, index):
        # List of lists of command lines (pipes)
        self.args = args
        # Location of golden files (out, err)
        self.golden = golden + '.' + repr(index)
        self.golden_error = golden + '.' + repr(index) + '.err'
        # Final output and error
        self.stdout = list()
        self.stderr = list()

    """Run a single command line, piping the output to the next command,
       if any, or to the final output, if none"""
    def _run(self, cmd, stdin):
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate(memoryview(stdin))
        # We combine stderr from all sub-processes into one
        if err:
            self.stderr.append(err.decode())
        # Return the CompletedProcess
        return process.returncode, out

    """Run the command lines, returning the exit status of the last
       piped command"""
    def run(self):
        stdin = b''
        # For each command line in args (list of lists)
        for cmd in self.args:
            # Run the program, accumulating stderr and piping stdout
            result, out = self._run(cmd, stdin)
            # On error, bail
            if result:
                return result
            # Otherwise, pass output to input
            stdin = out

        # Write final output to output file
        self.stdout = stdin.decode()

        return 0

    """Compare the output with the golden file,
       returning the difference if any"""
    def diff(self):
        # Dump is a human readable list that combines out/err into one text
        out = list()
        err = list()

        # Check stdout
        if self.stdout:
            if not os.path.exists(self.golden):
                print("Invalid path to stdout file for comparison", self.golden)
                return
            with open(self.golden, 'r') as gold:
                expected = gold.readlines()
                split = self.stdout.splitlines(True)
                if not frozenset(split).intersection(expected):
                    d = difflib.Differ()
                    out = list(d.compare(expected, split))

        # Check stderr
        if self.stderr:
            if not os.path.exists(self.golden_error):
                print("Invalid path to stderr file for comparison", self.golden_error)
                return
            with open(self.golden_error, 'r') as gold:
                expected = gold.readlines()
                split = self.stderr.splitlines(True)
                if not frozenset(split).intersection(expected):
                    d = difflib.Differ()
                    err = list(d.compare(expected, split))

        # Set return status
        status = out or err
        return status, out, err

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
            print(test.name, "test", index, "RUN FAILED", runner.args)
            sys.stdout.writelines(runner.stderr)
            status = -1
            continue

        # Show the diff, if any
        failed, out, err = runner.diff()
        if failed:
            print(test.name, "test", index, "FAILED", runner.args)
            print("OUT:")
            sys.stdout.writelines(out)
            print("ERR:")
            sys.stdout.writelines(err)
            status = -1
        else:
            print(test.name, "test", index, "PASSED")

        # Increment to next RUN line
        index += 1

    # Return non-zero on any error
    sys.exit(status)
