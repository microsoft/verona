// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "../src/host_service_calls.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

#include <limits>
#include <stdio.h>

using namespace sandbox;

HostServiceResponse sendRequest(HostServiceRequest req)
{
  auto written_bytes = write(PageMapUpdates, &req, sizeof(req));
  SANDBOX_INVARIANT(
    written_bytes == sizeof(req),
    "Wrote {} bytes, expected {}",
    written_bytes,
    sizeof(req));
  HostServiceResponse response;
  auto read_bytes = read(PageMapUpdates, &response, sizeof(response));
  SANDBOX_INVARIANT(
    read_bytes == sizeof(response),
    "Read {} bytes, expected {}",
    read_bytes,
    sizeof(response));
  return response;
}

uintptr_t try_dealloc(const void* addr, size_t size)
{
  HostServiceRequest req{
    DeallocChunk,
    {reinterpret_cast<uintptr_t>(addr), static_cast<uintptr_t>(size), 0, 0}};
  return sendRequest(req).error;
}

void try_alloc(const void* base)
{
  auto* header =
    static_cast<sandbox::SharedMemoryRegion*>(const_cast<void*>(base));
  SANDBOX_INVARIANT(
    sendRequest({AllocChunk, {0x100, 0, 0}}).error != 0,
    "Allocating less than a chunk spuriously returned success");
  uintptr_t ras =
    snmalloc::FrontendMetaEntry<snmalloc::FrontendSlabMetadata>::encode(
      &header->allocator_state,
      snmalloc::sizeclass_t::from_small_class(
        snmalloc::size_to_sizeclass_const(snmalloc::MIN_CHUNK_SIZE)));
  snmalloc::message<>("Trying to allocate parent-owned memory");
  SANDBOX_INVARIANT(
    sendRequest({AllocChunk, {snmalloc::MIN_CHUNK_SIZE, 4096, ras}}).error != 0,
    "Allocating a chunk owned by the parent spuriously returned success");
}

bool attack(const void* base, const void* top)
{
  try_alloc(base);
  SANDBOX_INVARIANT(try_dealloc(nullptr, -1) != 0, "Trying to dealloc nullptr");
  SANDBOX_INVARIANT(
    try_dealloc(static_cast<const char*>(base) + 100, -100) != 0,
    "Trying to dealloc with overflow");
  SANDBOX_INVARIANT(
    try_dealloc(static_cast<const char*>(base) + 100, 0xfffffffffffffff0) != 0,
    "Trying to dealloc with overflow 2");
#ifdef NDEBUG
  // This test triggers an assert failure in snmalloc in debug builds.  In
  // release builds, we should have proper error handling for it..
  SANDBOX_INVARIANT(
    try_dealloc(
      static_cast<const char*>(top) + 100,
      std::numeric_limits<size_t>::max() - 100) != 0,
    "Trying to dealloc with overflow from the top");
#else
  snmalloc::UNUSED(top);
#endif
  return false;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::attack);
}
