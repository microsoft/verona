// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <stdio.h>

using namespace sandbox;

bool attack(const void*, const void*);

/**
 * The structure that represents an instance of the sandbox.
 */
struct EvilSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(public_name, private_name) \
  decltype(make_sandboxed_function<decltype(private_name)>(lib)) public_name = \
    make_sandboxed_function<decltype(private_name)>(lib);
  EXPORTED_FUNCTION(attack, ::attack)
};

int main()
{
  EvilSandbox sandbox;
  auto [base, top] = sandbox.lib.sandbox_heap();
  SANDBOX_INVARIANT(
    sandbox.attack(base, top) == false, "Sandbox attack succeeded!");
  return 0;
}
