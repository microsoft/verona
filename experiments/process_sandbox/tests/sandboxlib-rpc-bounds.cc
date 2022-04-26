// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "../src/host_service_calls.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

#include <limits>
#include <stdio.h>

using namespace sandbox;
using namespace snmalloc;

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
  uintptr_t ras = FrontendMetaEntry<FrontendSlabMetadata>::encode(
    &header->allocator_state,
    sizeclass_t::from_small_class(size_to_sizeclass_const(MIN_CHUNK_SIZE)));
  message<>("Trying to allocate parent-owned memory");
  SANDBOX_INVARIANT(
    sendRequest({AllocChunk, {MIN_CHUNK_SIZE, 4096, ras}}).error != 0,
    "Allocating a chunk owned by the parent spuriously returned success");
}

/**
 * Attack from GitHub issue 565
 */
bool attack565(const void* base, const void* top)
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
  UNUSED(top);
#endif
  return false;
}

bool attack574(const void*, const void*)
{
  int error = 0;
  auto resp = sendRequest({AllocChunk, {MIN_CHUNK_SIZE, 0, 0}});
  if (resp.error != 0)
  {
    message("failed, error == {}", resp.error);
    // Allocation should not fail here
    return true;
  }
  char* p = reinterpret_cast<char*>(resp.ret);
  // parent segfault here before sending a valid resp.
  // p contains a pointer to a middle of some chunk, and we can use this address
  // of course, instead of doing so, we can simply free p+0x100, and we get a
  // similar segfault
  message("Freeing {} + 0x100", p);
  error = try_dealloc(p + 0x100, snmalloc::MIN_CHUNK_SIZE);
  if (error == 0)
  {
    message("Freeing misaligned chunk succeeded");
    return true;
  }
  message("Dealloc failed, error == {}", error);
  return false;
}

bool attack575(const void*, const void*)
{
  int error = 0;
  auto resp = sendRequest({AllocChunk, {MIN_CHUNK_SIZE, 0, 0}});
  if (resp.error != 0)
  {
    message("failed, error == {}", resp.error);
    // Allocation should not fail here
    return true;
  }
  char* p = reinterpret_cast<char*>(resp.ret);
  message("Freeing {}", p);
  error = try_dealloc(p, snmalloc::MIN_CHUNK_SIZE);
  if (error != 0)
  {
    message("Freeing {} failed", p);
    return true;
  }
  message("Freeing {}", p);
  error = try_dealloc(p, snmalloc::MIN_CHUNK_SIZE);
  if (error == 0)
  {
    message("Freeing {} for the second time succeeded", p);
    return true;
  }
  return false;
}

bool attack576(const void*, const void*)
{
  int error = 0;
  auto resp = sendRequest({AllocChunk, {MIN_CHUNK_SIZE * 2, 0, 0}});
  if (resp.error != 0)
  {
    message("failed, error == {}", resp.error);
    // Allocation should not fail here
    return true;
  }
  char* p = reinterpret_cast<char*>(resp.ret);
  message("Allocated chunk: {} bytes from {}", MIN_CHUNK_SIZE * 2, p);
  // parent segfault here before sending a valid resp.
  // p contains a pointer to a middle of some chunk, and we can use this address
  // of course, instead of doing so, we can simply free p+0x100, and we get a
  // similar segfault
  message("Freeing {} + 0x100", p);
  message(
    "Freeing too-small chunk, {} bytes from {}",
    snmalloc::MIN_CHUNK_SIZE - 100,
    p);
  error = try_dealloc(p, snmalloc::MIN_CHUNK_SIZE - 100);
  if (error == 0)
  {
    message("Freeing too-small chunk succeeded");
    return true;
  }
  message("Dealloc failed, error == {}", error);
  message("Freeing the second half of the allocation");
  error = try_dealloc(p + snmalloc::MIN_CHUNK_SIZE, snmalloc::MIN_CHUNK_SIZE);
  if (error != 0)
  {
    message("Freeing the second slab failed: {}", error);
    return true;
  }
  message(
    "Freeing too-large chunk, {} bytes from {}",
    snmalloc::MIN_CHUNK_SIZE * 2,
    p);
  error = try_dealloc(p, snmalloc::MIN_CHUNK_SIZE * 2);
  if (error == 0)
  {
    message("Freeing too-large chunk succeeded");
    return true;
  }
  message("Dealloc failed, error == {}", error);
  return false;
}

static bool attack(int issue, const void* base, const void* top)
{
  switch (issue)
  {
    case 565:
      return attack565(base, top);
    case 574:
      return attack574(base, top);
    case 575:
      return attack575(base, top);
    case 576:
      return attack576(base, top);
  }
  // Invalid PR number, pretend that the attack succeeded so that we hit an
  // assert fail in the parent.
  return true;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::attack);
}
