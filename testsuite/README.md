# Verona testsuite
## Layout
Test cases are divided in top-level directories, grouped by feature they relate
to. Within each top-level directory, tests are further split by _mode_.

## Test modes
Each test's expectations depends on its mode.
- `compile-pass`: Compilation must succeed.
- `compile-fail`: Compilation must fail. The compiler's standard error will be
  compared against the test file using `OutputCheck`.

Each mode is implemented by a `.cmake` file at the top of the testsuite
directory.

## Checking dump files

Regardless of the mode, if dump files are provided, they will be compared
against the ones produce by the compiler. The expected outputs should be placed
as text files, in a directory with the same name as the test case. For example
the expected `ast` output for `foo/compile-pass/bar.verona` would be located in
`foo/compile-pass/bar/ast.txt`.

Any file dumped by the compiler for which no corresponding file exists in the
testsuite will be ignored.

If the output of the compiler changes, causing it to differ with the expected
results, the testsuite will fail to pass. The expected outputs can be updated to
reflect the compiler's output by running `ninja update-dump`.
