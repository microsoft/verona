
Project Verona is a research programming language to explore the concept of
concurrent ownership.  We are providing a new concurrency model that seamlessly
integrates ownership.

This research project is at an early stage and is open sourced to facilitate 
academic collaborations.  We are keen to engage in research collaborations on
this project, please do reach out to discuss this.

The project is not ready to be used outside of research.

# Status

This project is at a very early stage, parts of the type checker are still to be
implemented, and there are very few language features implemented yet. This will
change, but will take time.

## Nightly Build

Distro\Build | Release | Debug | ASAN 
--------|---------|-------|--------
Linux | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=Linux%20(Verona)&jobName=Linux&configuration=Linux%20Clang%20Release)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=Linux%20(Verona)&jobName=Linux&configuration=Linux%20Clang%20Debug)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=Linux%20(Verona)&jobName=Linux&configuration=Linux%20Clang%20Debug%20(ASAN))](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master)
Windows | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=Windows%20(Verona)&jobName=Windows&configuration=Windows%20Release)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=Windows%20(Verona)&jobName=Windows&configuration=Windows%20RelWithDebInfo)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) |
Darwin | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=macOS%20(Verona)&jobName=macOS&configuration=macOS%20Release)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) | [![Build Status](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_apis/build/status/Verona%20Nightly?branchName=master&stageName=macOS%20(Verona)&jobName=macOS&configuration=macOS%20Debug)](https://dev.azure.com/ProjectVeronaCI/Project%20Verona/_build/latest?definitionId=11&branchName=master) |

# Syntax Highlight

Due to some amazing early contributors, we have syntax highlighting for two editors:
* **VS Code:** Automatically suggests installing when opening Verona files. Thanks to @KodrAus (#145).
* **Emacs:** https://github.com/damon-kwok/verona-mode. Thanks to @damon-kwok (#199).

**Vim** lovers, the ball is in your court.

# Development Documents

## [Exploring the language](docs/explore.md)

## [FAQ](docs/faq.md)

## [Building](docs/building.md)

## [Contributing](CONTRIBUTING.md)
