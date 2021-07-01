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

template<typename P>
void test_poller()
{
  P poller;
  std::unordered_set<int> read_fds;
  int closed_fds = 0;
  static constexpr int fd_count = 5;
  std::atomic<int> read_fd_count{0};
  std::thread t([&]() {
    while (closed_fds < fd_count)
    {
      handle_t h;
      bool eof;
      bool ret = poller.poll(h, eof);
      SANDBOX_INVARIANT(ret, "Poller returned failure");
      int val = -1;
      if (read(h, &val, sizeof(val)) > 0)
      {
        read_fds.insert(val);
        read_fd_count++;
      }
      if (eof)
      {
        closed_fds++;
      }
    }
  });
  {
    std::vector<Handle> socks;
    for (int i = 0; i < fd_count; i++)
    {
      auto sp = SocketPair::create();
      poller.add(sp.second.take());
      int ret = write(sp.first.fd, &i, sizeof(i));
      SANDBOX_INVARIANT(ret == sizeof(i), "write failed {}", strerror(errno));
      socks.push_back(std::move(sp.first));
    }
    auto start = std::chrono::steady_clock::now();
    auto timeout = start + std::chrono::seconds(timeout_seconds);
    while (read_fd_count < fd_count)
    {
      SANDBOX_INVARIANT(
        std::chrono::steady_clock::now() < timeout, "Test timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  auto future = std::async(std::launch::async, &std::thread::join, &t);
  // Join or time out after 5 seconds so the test fails if we infinite loop
  SANDBOX_INVARIANT(
    future.wait_for(std::chrono::seconds(timeout_seconds)) !=
      std::future_status::timeout,
    "Test timed out");
  SANDBOX_INVARIANT(
    closed_fds == fd_count,
    "{} fds were closed, {} expected",
    closed_fds,
    fd_count);
  for (int i = 0; i < fd_count; i++)
  {
    SANDBOX_INVARIANT(
      read_fds.count(i) == 1,
      "fd {} was read {} times, once was expected",
      i,
      read_fds.count(i));
  }
}

int main(void)
{
  test_poller<Poller>();
}
