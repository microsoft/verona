// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>

int sum(int a, int b)
{
  fprintf(stderr, "Adding %d to %d in sandbox\n", a, b);
  return a + b;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::sum);
}
