// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "cxxapi/cxxsandbox.h"
#include "sandbox.hh"
#include "shared.h"

int crash()
{
  abort();
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::crash);
}
