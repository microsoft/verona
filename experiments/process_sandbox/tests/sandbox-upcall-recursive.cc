// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/filetree.h"
#include "process_sandbox/privilege_elevation_upcalls.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>

using namespace sandbox;

/**
 * The structure that represents an instance of the sandbox.
 */
struct UpcallSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  SandboxedLibrary lib = {SANDBOX_LIBRARY};
  decltype(make_sandboxed_function<int(int, int)>(lib)) call_upcall =
    make_sandboxed_function<int(int, int)>(lib);
};

int main()
{
  UpcallSandbox sandbox;
  int upcall_number;
  int expected = 11;
  auto upcall = [&](SandboxedLibrary&, int val) {
    fprintf(stderr, "[h] Upcall invoked %d\n", val);
    UpcallHandlerBase::Result r{0};
    assert(val == expected);
    expected -= 2;
    if (val > 0)
    {
      r = sandbox.call_upcall(upcall_number, val - 1);
      fprintf(stderr, "[h] Upcall %d returned\n", val);
    }
    return r;
  };
  upcall_number =
    sandbox.lib.register_callback(sandbox::make_upcall_handler<int>(upcall));
  try
  {
    fprintf(stderr, "Asking sandbox to invoke callback %d\n", upcall_number);
    int ret = sandbox.call_upcall(upcall_number, expected + 1);
    fprintf(stderr, "Sandbox returned %d\n", ret);
    assert(ret == 0);
  }
  catch (...)
  {
    assert(0);
  }
  return 0;
}
