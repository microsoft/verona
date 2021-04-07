// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "process_sandbox/helpers.h"
#include "process_sandbox/platform/platform.h"

#include <thread>

using namespace sandbox::platform;

template<typename Child>
void test_child()
{
  Child cp([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    exit(2);
  });
  auto ec = cp.wait_for_exit();
  SANDBOX_INVARIANT(ec.has_exited, "Child has not exited");
  SANDBOX_INVARIANT(
    ec.exit_code == 2, "Error code is {}, expected 2", ec.exit_code);
  auto ec2 = cp.exit_status();
  SANDBOX_INVARIANT(ec2.has_exited, "Child has not exited");
  SANDBOX_INVARIANT(
    ec2.exit_code == 2, "Error code is {}, expected 2", ec.exit_code);
}

using Fallback =
#ifdef __unix__
  ChildProcessVFork
#else
  ChildProcess
#endif
  ;

int main(void)
{
  test_child<ChildProcess>();
  if (!std::is_same_v<ChildProcess, Fallback>)
  {
    test_child<Fallback>();
  }
}
