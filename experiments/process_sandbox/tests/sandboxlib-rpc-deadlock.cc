// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "../src/host_service_calls.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

#include <limits>
#include <stdio.h>

using namespace sandbox;
using snmalloc::UNUSED;

void attack()
{
  HostServiceRequest req{AllocChunk, {0, 0, 0}};

  // Keep writing nonsense values, without reading.  This should fill up the
  // response buffer in the kernel and cause writes from the parent to fail.
  // This sandbox should be terminated if that happens.
  for (int i = 0; i < 1000000; i++)
  {
    UNUSED(write(PageMapUpdates, &req, sizeof(req)));
  }
}

void attack2()
{
  // Try slowly writing individual bytes.  This should cause the parent to
  // detect partial messages and kill the sandbox.
  for (int i = 0; i < 100; i++)
  {
    UNUSED(write(PageMapUpdates, "\0", 1));
    usleep(100000);
  }
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::attack);
  sandbox::ExportedLibrary::export_function(::attack2);
}
