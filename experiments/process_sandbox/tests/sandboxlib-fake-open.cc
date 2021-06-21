// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/platform/platform.h"
#include "process_sandbox/sandbox.h"

#include <sys/syscall.h>
#include <unistd.h>

using SyscallFrame = sandbox::platform::SyscallFrame;

int test(bool raw_syscall)
{
  const char* path = "/foo";
  int x;
  if (raw_syscall && (SyscallFrame::Open != -1))
  {
    x = syscall(SyscallFrame::Open, path, O_RDWR);
  }
  else
  {
    x = open(path, O_RDWR);
  }
  // Make sure that we're writing to the start of the 'file'.
  lseek(x, SEEK_SET, 0);
  char buf[] = "hello world";
  int bytes = write(x, buf, sizeof(buf));
  close(x);
  return bytes;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::test);
}
