Clang interface experiment
==========================

This experiment is a quick-and-dirty hack to explore whether we can use clang to:

 - Parse a C/C++ header.
 - Cache the parse result so that the Verona compiler can do incremental builds.
 - Resolve types and symbols.
 - Instantiate templates.
 - Synthesize functions that provide simple-ABI wrappers.

It is incomplete and ugly code but serves to demonstrate that the clang APIs are sufficient for our purposes.
