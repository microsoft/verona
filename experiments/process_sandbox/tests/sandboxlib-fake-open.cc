// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/callback_numbers.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/platform/platform.h"
#include "process_sandbox/sandbox.h"

#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

using SyscallFrame = sandbox::platform::SyscallFrame;

int test(bool raw_syscall)
{
  // Disable the raw syscall test on FreeBSD versions that can't support it.
  // This is currently anything <14 but hopefully the code will be MFC'd to
  // other releases soon, at which point this can be deleted.
#if defined(__FreeBSD__) && !defined(si_syscall)
  raw_syscall = false;
#endif
  const char* path = "/foo";
  int x;
  constexpr int OpenSyscallNo =
    SyscallFrame::syscall_number<sandbox::CallbackKind::Open>();
  if (raw_syscall && (OpenSyscallNo != -1))
  {
    x = syscall(OpenSyscallNo, path, O_RDWR);
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
