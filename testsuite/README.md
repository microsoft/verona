# Testsuite

The shared testsuite registers named CTest DAGs from top-level collection
files:

| Collection | Pipeline |
| --- | --- |
| `vc-vbc.cmake` | Verona source -> VC -> VBC -> VBCI |
| `virc-vbc.cmake` | Textual VIR -> VIRC -> VBC -> VBCI |

These collection names identify both the selected input and VBC output path.

Every node has `exit_code.txt`, `stdout.txt`, and `stderr.txt` goldens; silent
files are empty and exit-code files have no trailing newline.

From the repository root, use `cmake --build --preset debug` to build and
install, then `ctest --preset debug` to verify goldens. The `release` presets
use a separate build and installation tree. Use `ninja -C build/debug
update-dump` only to intentionally regenerate goldens.

Detailed guidance:

- [Pipelines](docs/pipelines.md)
- [Fixtures](docs/fixtures.md)
- [Golden files](docs/golden-files.md)