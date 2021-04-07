// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/callbacks.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/filetree.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>

using namespace sandbox;

/**
 * The structure that represents an instance of the sandbox.
 */
struct CallbackSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
  decltype(make_sandboxed_function<int(int)>(lib)) call_callback =
    make_sandboxed_function<int(int)>(lib);
};

CallbackHandlerBase::Result callback(Library&, int val)
{
  // 12 is an arbitrary number, used by the caller of this in the other file.
  SANDBOX_INVARIANT(val == 12, "Callback argument is {}, expected 12", val);
  return 42;
}

int main()
{
  CallbackSandbox sandbox;
  int callback_number = sandbox.lib.register_callback(
    sandbox::make_callback_handler<int>(callback));
  try
  {
    int ret = sandbox.call_callback(callback_number);
    SANDBOX_INVARIANT(ret == 42, "Sandbox returned {}, expected 42", ret);
  }
  catch (...)
  {
    SANDBOX_INVARIANT(0, "Exception thrown when invoking sandbox");
  }
  return 0;
}
