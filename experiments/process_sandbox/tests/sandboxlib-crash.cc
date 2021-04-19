// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

int crash()
{
  abort();
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::crash);
}
