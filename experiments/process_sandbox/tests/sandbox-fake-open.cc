// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "cxxapi/cxxsandbox.h"
#include "filetree.h"
#include "sandbox.hh"
#include "shared.h"

#include <stdio.h>

using namespace sandbox;

/**
 * The structure that represents an instance of the sandbox.
 */
struct VFSSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  SandboxedLibrary lib = {SANDBOX_LIBRARY};
  decltype(make_sandboxed_function<int(bool)>(lib)) test =
    make_sandboxed_function<int(bool)>(lib);
};

int main()
{
  VFSSandbox sandbox;
  platform::SharedMemoryMap fake_file(16);
  sandbox.lib.filetree().add_file("/foo", std::move(fake_file.get_handle()));
  char expected[] = "hello world";
  auto test = [&](bool raw_syscall) {
    memset(fake_file.get_base(), 0, sizeof(expected));
    try
    {
      int ret = sandbox.test(raw_syscall);
      fprintf(stderr, "Sandbox returned %d\n", ret);
      assert(ret == sizeof(expected));
    }
    catch (...)
    {
      assert(0);
    }
    assert(memcmp(fake_file.get_base(), expected, sizeof(expected)) == 0);
  };
  fprintf(stderr, "Indirect syscall\n");
  test(false);
  fprintf(stderr, "Direct syscall\n");
  test(true);
  return 0;
}
