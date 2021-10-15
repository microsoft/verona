// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <stdio.h>

using namespace sandbox;

/**
 * Function that will try to escape from the sandbox.  Returns the result of
 * trying to make the shared pagemap read-write.
 */
int attack();

/**
 * The structure that represents an instance of the sandbox.
 */
struct BadSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(name) \
  decltype(make_sandboxed_function<decltype(::name)>(lib)) name = \
    make_sandboxed_function<decltype(::name)>(lib);
  EXPORTED_FUNCTION(attack)
};

int main()
{
  BadSandbox sandbox;
  // attack() will try to make the pagemap read-write.
  SANDBOX_INVARIANT(sandbox.attack() != 0, "Sandbox attack failed");
  return 0;
}
