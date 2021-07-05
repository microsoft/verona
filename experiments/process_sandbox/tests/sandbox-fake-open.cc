// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/filetree.h"
#include "process_sandbox/sandbox.h"

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
  Library lib = {SANDBOX_LIBRARY};
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
  // The no-op sandbox doesn't actually do any sandboxing so raw system calls
  // will work.  Skip the test that they're correctly intercepted.
  if constexpr (!std::is_same_v<platform::Sandbox, platform::SandboxNoOp>)
  {
    fprintf(stderr, "Direct syscall\n");
    test(true);
  }
  return 0;
}
