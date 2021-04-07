// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "process_sandbox/helpers.h"
#include "process_sandbox/platform/platform.h"

#include <future>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace sandbox::platform;

// Timeout.  If you are debugging this test, increase this so that it doesn't
// fail in the background while you're inspecting a breakpoint.
constexpr int timeout_seconds = 5;

template<typename Sem>
void test_sem()
{
  SharedMemoryMap map(16);
  std::atomic<bool> passed{false};
  auto* sem = new (map.get_base()) Sem();

  // Spawn another thread spawns a child process that waits with a long
  // timeout.  We spawn the child process in a new thread because there's no
  // requirement that the child is executed in parallel until it calls execve
  // (which doesn't happen here).
  std::thread t([&]() {
    ChildProcess p([&]() { exit(sem->wait(timeout_seconds * 1000)); });
    auto ret = p.wait_for_exit();
    SANDBOX_INVARIANT(
      ret.exit_code == 1, "Exit code is {}, expected 1", ret.exit_code);
    passed = true;
  });
  sem->wake();

  auto future = std::async(std::launch::async, &std::thread::join, &t);
  // Join or time out after 5 seconds so the test fails if we infinite loop
  SANDBOX_INVARIANT(
    future.wait_for(std::chrono::seconds(timeout_seconds)) !=
      std::future_status::timeout,
    "Test timed out");
  SANDBOX_INVARIANT(passed, "Wake did not occur");
}

int main(void)
{
  test_sem<OneBitSem>();
}
