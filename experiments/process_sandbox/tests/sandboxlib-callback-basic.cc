// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

int test(int idx)
{
  int v = 12;
  int ret = sandbox::invoke_user_callback(idx, &v, sizeof(v));
  assert(ret == 42);
  return ret;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::test);
}
