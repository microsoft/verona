// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "process_sandbox/helpers.h"
#include "process_sandbox/platform/platform.h"

using namespace sandbox::platform;

template<typename Map>
void test_map()
{
  // Pick a fairly small size so we won't exhaust memory on a CI VM.
  size_t log2size = 20;
  size_t size = 1 << log2size;
  uintptr_t address_mask = size - 1;
  // Construct the shared memory object.
  Map m(log2size);
  // Is the base correctly aligned?
  uintptr_t base = reinterpret_cast<uintptr_t>(m.get_base());
  SANDBOX_INVARIANT(
    (base & address_mask) == 0,
    "Base address {} is insufficiently aligned",
    base);
  // Is the size what we asked for?
  SANDBOX_INVARIANT(
    m.get_size() == size, "Mapped object is {} bytes", m.get_size());
  // Can we at least write to and read from the first and last byte?
  auto cp = static_cast<volatile char*>(m.get_base());
  cp[0] = 12;
  cp[size - 1] = 42;
  SANDBOX_INVARIANT(
    cp[0] == 12,
    "Value 12 stored at the start of the map, read back as {:d}",
    cp[0]);
  SANDBOX_INVARIANT(
    cp[size - 1] == 42,
    "Value 42 stored at the end of the map, read back as {:d}",
    cp[size - 1]);
}

using FallbackMap =
#ifdef __unix__
  SharedMemoryMapPOSIX<detail::SharedMemoryObjectPOSIX>
#else
  SharedMemoryMap
#endif
  ;

int main(void)
{
  test_map<SharedMemoryMap>();
  // If we are using a specialised version, also test the portable version
  if constexpr (!std::is_same_v<FallbackMap, SharedMemoryMap>)
  {
    test_map<FallbackMap>();
  }
}
