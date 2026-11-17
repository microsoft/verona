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

Run `ninja update-dump` from `build/` only when intentionally regenerating
goldens. A normal validation uses `ninja install` and `ctest`; it must not modify
expected output.