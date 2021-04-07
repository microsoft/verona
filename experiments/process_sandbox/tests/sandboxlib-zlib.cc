// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <zlib.h>

extern "C" void sandbox_init()
{
#define EXPORTED_FUNCTION(x, name) \
  sandbox::ExportedLibrary::export_function(name);
#include "zlib.inc"
}
