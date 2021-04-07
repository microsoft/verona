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
  Sem sem;
  std::atomic<bool> passed{0};
  // Check that we time out without acquiring the semaphore.
  bool acquired = sem.wait(100);
  SANDBOX_INVARIANT(!acquired, "Failed to acquire semaphore");
  // Spawn another thread that waits with a long timeout.
  std::thread t([&]() {
    sem.wait(timeout_seconds * 1000);
    passed = true;
  });
  sem.wake();

  auto future = std::async(std::launch::async, &std::thread::join, &t);
  // Join or time out after 5 seconds so the test fails if we infinite loop
  SANDBOX_INVARIANT(
    future.wait_for(std::chrono::seconds(timeout_seconds)) !=
      std::future_status::timeout,
    "Test timed out");
  SANDBOX_INVARIANT(passed, "Wait never woke up");
}

int main(void)
{
  test_sem<OneBitSem>();
}
