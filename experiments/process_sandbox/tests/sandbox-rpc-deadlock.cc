// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <stdio.h>

using namespace sandbox;

void attack();
void attack2();

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
  EXPORTED_FUNCTION(attack2, ::attack2)
};

void attack()
{
  EvilSandbox sandbox;
  try
  {
    sandbox.attack();
  }
  catch (...)
  {
    return;
  }
  SANDBOX_INVARIANT(0, "First attack succeeded");
}

void attack2()
{
  EvilSandbox sandbox;
  try
  {
    sandbox.attack2();
  }
  catch (...)
  {
    return;
  }
  SANDBOX_INVARIANT(0, "Second attack succeeded");
}

int main()
{
  attack();
  attack2();
}
