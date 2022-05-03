// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <stdio.h>

using namespace sandbox;

bool attack(int, const void*, const void*);

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

void try_attack(int issue)
{
  EvilSandbox sandbox;
  auto [base, top] = sandbox.lib.sandbox_heap();
  try
  {
    SANDBOX_INVARIANT(
      sandbox.attack(issue, base, top) == false,
      "Sandbox attack {} succeeded!",
      issue);
  }
  catch (...)
  {}
}

int main()
{
  try_attack(565);
  try_attack(574);
  try_attack(575);
  try_attack(576);
  return 0;
}
