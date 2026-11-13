# Contributing

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Working on Verona

[TODO]

## Style

We use `clang-format-20` on our CI bots, which will fail the pull request if the code is not formatted as expected.

The `clang-format` tool doesn't guarantee backward or forward compatibility, so we have to fix a version for our CI and local development.

Install version 20 for local development; other versions may produce incompatible formatting.

Install it from the [LLVM releases](https://releases.llvm.org/) or your system package manager, then put `clang-format-20` on your PATH *before* you run CMake.

To format Verona's C++ sources and headers:

```sh
cmake --build build --target verona-clangformat
```

To check formatting without modifying files:

```sh
cmake --build build --target verona-clangformat-check
```

These targets exclude fetched dependencies and build outputs. CI runs the non-mutating check.

# Bugs and patches

We use Github [issues](https://github.com/microsoft/verona/issues) to track bugs and features. If you found a bug or want to propose a new feature, please file an issue on our project.

We use Github [pull requests](https://github.com/microsoft/verona/pulls) for contributions in code, documents, tests, etc. 

Every PR must pass the CI builds (which include CLA checks and formatting) and the appropriate set of tests on Windows, Linux (clang & gcc) and Mac, on x86_64. PRs cannot be merged if any of the tests fail.

You are not, however, required to run all these tests on your own, before submitting the PR. Running on at least one of those above and passing should be fine. We can work out the remaining issues during the review process.
