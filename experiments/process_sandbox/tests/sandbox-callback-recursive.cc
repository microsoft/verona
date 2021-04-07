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
  decltype(make_sandboxed_function<int(int, int)>(lib)) call_callback =
    make_sandboxed_function<int(int, int)>(lib);
};

int main()
{
  CallbackSandbox sandbox;
  int callback_number;
  int expected = 11;
  auto callback = [&](Library&, int val) {
    fprintf(stderr, "[h] Callback invoked %d\n", val);
    CallbackHandlerBase::Result r{0};
    assert(val == expected);
    expected -= 2;
    if (val > 0)
    {
      r = sandbox.call_callback(callback_number, val - 1);
      fprintf(stderr, "[h] Callback %d returned\n", val);
    }
    return r;
  };
  callback_number = sandbox.lib.register_callback(
    sandbox::make_callback_handler<int>(callback));
  try
  {
    fprintf(stderr, "Asking sandbox to invoke callback %d\n", callback_number);
    int ret = sandbox.call_callback(callback_number, expected + 1);
    fprintf(stderr, "Sandbox returned %d\n", ret);
    assert(ret == 0);
  }
  catch (...)
  {
    assert(0);
  }
  return 0;
}
