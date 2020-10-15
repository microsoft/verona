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

See the [building](docs/building.md) document for more information on how to get and build Verona on multiple operating systems.

When submiting patches, you need to make sure all tests pass (`check` target) and that the code is formatted accordingly.

We provide the `compiler_commands.json` file for editors that use the [clangd](https://clangd.llvm.org/) engine for code completion, and we `git-ignore` meta files from common editors, like VSCode and Vim.

## Tests

If you're only changing the compiler, interpreter or parser, then running the `check` target is enough to cover those areas. On Windows, there is no separate `check` target, and the tests are run as part of the default target.

If you're changing the runtime libraries, then you'll need to [build the RT tests](docs/building.md#building-the-runtime-tests) and run them before submiting a PR.

Our CI bots will check if you changed the runtime and will dispatch the runtime tests automatically.

## Style

We use `clang-format-9` on our CI bots, which will fail the pull request if the code is not formatted as expected.

The `clang-format` tool doesn't guarantee backward or forward compatibility, so we have to fix a version for our CI and local development.

Both versions 8 and 10 (and therefore any other version) is incompatible with version 9 and *will* break the build, so you must download and use version 9.

To do so, you can choose a number of ways to download it (ex. [npm](https://www.npmjs.com/package/clang-format), [releases](https://releases.llvm.org/), [apt](https://packages.ubuntu.com/search?suite=default&section=all&arch=any&keywords=clang-format-9&searchon=names), [Arch AUR](https://aur.archlinux.org/packages/clang-format-static-bin/), etc), then put on your PATH *before* you run CMake.

A `clangformat` target will be created and you can make sure you won't break the build by running that target.

# Bugs and patches

We use Github [issues](https://github.com/microsoft/verona/issues) to track bugs and features. If you found a bug or want to propose a new feature, please file an issue on our project.

We use Github [pull requests](https://github.com/microsoft/verona/pulls) for contributions in code, documents, tests, etc. 

Every PR must pass the CI builds (which include CLA checks and formatting) and the appropriate set of tests on Windows, Linux (clang & gcc) and Mac, on x86_64. PRs cannot be merged if any of the tests fail.

You are not, however, required to run all these tests on your own, before submitting the PR. Running on at least one of those above and passing should be fine. We can work out the remaining issues during the review process.
