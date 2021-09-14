// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../pal/threading.h"

#include <condition_variable>

namespace verona::rt
{
  /**
   * This class encapsulate the logic for allowing a separate
   * thread to be notified to execute some supplied code.
   *
   * This is intended for use during signals so that complex code
   * can be executed in response to a signal, but only calls a very
   * small number of things to cause that to happen. `ping` is intended
   * to be signal safe, and causes the encapsulated thread to execute
   * the required code asynchronously.
   */
  template<typename Fn>
  class ThreadPing
  {
    /// How many pings have been requested.
    /// Must hold mutx to access
    size_t requests = 0;

    /// Specifies if we are in the destructor. Used to communicate
    /// internal thread should terminate.
    /// Must hold mutx to access
    bool teardown = false;

    /// Used by thrd, to wait for more notifications.
    /// Must hold mutx to access
    std::condition_variable cv{};

    /// Protects internal start of the Ping
    std::mutex mutx{};

    /// Internal thread that will handle the pings.
    PlatformThread thrd;

  public:
    /// Creates a thread that can be pinged to execute f
    /// asynchronously.
    ThreadPing(Fn&& f)
    : thrd(([&, f = std::move(f)]() {
        size_t serving = 0;
        std::unique_lock<std::mutex> lock(mutx);
        while (true)
        {
          // Wait for a new notification either a ping, or teardown due to
          // destruction.
          cv.wait(lock, [&] { return teardown || (serving < requests); });

          // Destructor has run, we need to finish.
          if (teardown)
            return;

          // Run payload
          f();
          serving++;
        }
      }))
    {}

    /// Cause internal thread to execute the specified closure f again.
    /// returns without waiting for the execution to have occurred.
    void ping()
    {
      std::unique_lock<std::mutex> lock(mutx);
      requests++;
      // Notify while holding the lock, assuming `wait morphing`
      // optimisation.
      cv.notify_one();
    }

    /// Join internal thread, and wait for it to terminate before
    /// tearing down the mutex and condition variable.
    ~ThreadPing()
    {
      {
        std::unique_lock<std::mutex> lock(mutx);
        teardown = true;
        // Notify while holding the lock, assuming `wait morphing`
        // optimisation.
        cv.notify_one();
      }
      thrd.join();
    }
  };
}