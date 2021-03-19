// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "cxxapi/cxxsandbox.h"
#include "filetree.h"
#include "privilege_elevation_upcalls.h"
#include "sandbox.hh"
#include "shared.h"

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
  decltype(make_sandboxed_function<int(int)>(lib)) call_upcall =
    make_sandboxed_function<int(int)>(lib);
};

UpcallHandlerBase::Result upcall(SandboxedLibrary&, int val)
{
  SANDBOX_INVARIANT(val == 12, "Upcall argument is {}, expected 12", val);
  return 42;
}

int main()
{
  UpcallSandbox sandbox;
  int upcall_number =
    sandbox.lib.register_callback(sandbox::make_upcall_handler<int>(upcall));
  try
  {
    int ret = sandbox.call_upcall(upcall_number);
    SANDBOX_INVARIANT(ret == 42, "Sandbox returned {}, expected 42", ret);
  }
  catch (...)
  {
    SANDBOX_INVARIANT(0, "Exception thrown when invoking sandbox");
  }
  return 0;
}
