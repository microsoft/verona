// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

int test(int idx, int count)
{
  fprintf(stderr, "[s] In sandbox %d\n", count);
  if (count > 0)
  {
    count--;
    fprintf(stderr, "[s] Invoking upcall: %d\n", count);
    return sandbox::invoke_user_callback(idx, &count, sizeof(count));
    fprintf(stderr, "[s] Upcall returned: %d\n", count);
  }
  return 0;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::test);
}
