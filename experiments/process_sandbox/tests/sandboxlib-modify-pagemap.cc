// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <dlfcn.h>
#include <stdio.h>

int attack()
{
  // Find the symbol from the library runner that contains the pagemap
  // address and load it.
  void* pagemap_base = *reinterpret_cast<void**>(
    dlsym(RTLD_DEFAULT, "_ZN7sandbox15SnmallocGlobals7Pagemap7pagemapE"));
  SANDBOX_INVARIANT(
    pagemap_base != nullptr,
    "The mangled name or the visibility of the child's sandbox has changed.  "
    "This test must be updated");
  fprintf(stderr, "Found pagemap base %p\n", pagemap_base);
  // Try to make the pagemap read-write.
  int ret =
    mprotect(pagemap_base, snmalloc::Pal::page_size, PROT_READ | PROT_WRITE);
  fprintf(stderr, "mprotect returned %d (%s)", ret, strerror(errno));
  // If we could, return 0, otherwise return the errno value.
  return ret == 0 ? 0 : errno;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::attack);
}
