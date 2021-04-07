// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>

using namespace sandbox;

int crash();

/**
 * The structure that represents an instance of the sandbox.
 */
struct CrashySandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(public_name, private_name) \
  decltype(make_sandboxed_function<decltype(private_name)>(lib)) public_name = \
    make_sandboxed_function<decltype(private_name)>(lib);
  EXPORTED_FUNCTION(crash, ::crash)
};

void test_crash(CrashySandbox& sb)
{
  try
  {
    printf(
      "Calling a function that should cause the sandboxed process to "
      "crash...\n");
    sb.crash();
  }
  catch (std::runtime_error& e)
  {
    printf("Sandbox exception: %s\n", e.what());
    printf("Parent process continuing happily...\n");
  }
}

int main()
{
  CrashySandbox sandbox;
  CrashySandbox sb2;
  CrashySandbox sb3;
  test_crash(sandbox);
  test_crash(sb2);
  test_crash(sb3);

  return 0;
}
