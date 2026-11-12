# Golden Files

Each registered command compares its process result with fixture goldens:

- `exit_code.txt` contains the decimal exit code with no trailing newline.
- `stdout.txt` contains expected standard output.
- `stderr.txt` contains expected standard error.
- silent output files are empty files, not absent files.

VC fixtures store compile and run expectations beneath their fixture output
directory. Compile-only failures use exit code `1` for the compile node and do
not define a run directory.

VIR fixtures can also provide `compile/<name>_final.trieste` beneath their
fixture output directory to compare the final optimized IR as a golden.
Fixtures without this file retain only the standard process-result checks.
Reconfigure CMake after adding or removing an optional IR golden.

Run `ninja -C build/debug update-dump` from the repository root only when
intentionally regenerating goldens. A normal validation uses
`cmake --build --preset debug` (including installation) and
`ctest --preset debug`; it must not modify expected output.

VC compile nodes normalize installation-dependent source-location prefixes
to `_builtin/` before comparison and updates. Builtin-relative filenames,
line/column numbers, diagnostic text, and exit codes remain checked. Do not
regenerate goldens just to change a developer's build path.
