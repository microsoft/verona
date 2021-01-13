// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "cxxapi/cxxsandbox.h"
#include "sandbox.hh"
#include "shared.h"

extern "C" void sandbox_init()
{
#define EXPORTED_FUNCTION(x, name) \
  sandbox::ExportedLibrary::export_function(name);
#include "zlib.inc"
}
