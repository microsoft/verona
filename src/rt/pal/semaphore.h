// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This file provides a mechanism for threads to sleep and be woken.
 *
 * To builds a point-to-point wake up using a binary semaphore. The
 * class is intentionally restricted to allow for other platforms to
 * implement this more efficiently.
 */
#ifndef VERONA_EXTERNAL_SEMAPHORE_IMPL
/**
 * This constructs a platform specific semaphore.
 */
#  if __has_include(<version>)
#    include <version>
#  endif
#  if defined(__cpp_lib_semaphore)
#    include <semaphore>
namespace verona::rt::pal
{
  class SemaphoreImpl
  {
    std::binary_semaphore semaphore_{0};

  public:
    void release()
    {
      semaphore_.release();
    }

    void acquire()
    {
      semaphore_.acquire();
    }
  };
} // namespace verona::rt::pal
#  elif defined(__APPLE__)
#    include <dispatch/dispatch.h>
namespace verona::rt::pal
{
  class SemaphoreImpl
  {
    dispatch_semaphore_t semaphore_;

  public:
    SemaphoreImpl()
    {
      semaphore_ = dispatch_semaphore_create(0);
    }

    ~SemaphoreImpl()
    {
      dispatch_release(semaphore_);
    }

    void release()
    {
      dispatch_semaphore_signal(semaphore_);
    }

    void acquire()
    {
      dispatch_semaphore_wait(semaphore_, DISPATCH_TIME_FOREVER);
    }
  };
} // namespace verona::rt::pal
#  elif defined(WIN32)
#    include <windows.h>
namespace verona::rt::pal
{
  class SemaphoreImpl
  {
    HANDLE semaphore_;

  public:
    SemaphoreImpl()
    {
      semaphore_ = CreateSemaphore(NULL, 0, LONG_MAX, NULL);
    }

    ~SemaphoreImpl()
    {
      CloseHandle(semaphore_);
    }

    void release()
    {
      ReleaseSemaphore(semaphore_, 1, NULL);
    }

    void acquire()
    {
      WaitForSingleObject(semaphore_, INFINITE);
    }
  };
} // namespace verona::rt::pal
#  elif __has_include(<semaphore.h>)
// Use Posix semaphores
#    include <semaphore.h>
namespace verona::rt::pal
{
  class SemaphoreImpl
  {
    sem_t semaphore_;

  public:
    SemaphoreImpl()
    {
      auto err = sem_init(&semaphore_, 0, 0);
      if (err != 0)
      {
        // Failed to initialize semaphore.
        abort();
      }
    }

    SemaphoreImpl(const SemaphoreImpl&) = delete;
    SemaphoreImpl& operator=(const SemaphoreImpl&) = delete;

    ~SemaphoreImpl()
    {
      sem_destroy(&semaphore_);
    }

    void release()
    {
      auto res = sem_post(&semaphore_);
      if (res != 0)
      {
        // Failed to release semaphore.
        abort();
      }
    }

    void acquire()
    {
      while (true)
      {
        auto err = sem_wait(&semaphore_);
        if (err == 0)
        {
          return;
        }
        else if (err == EINTR)
        {
          // Interrupted by a signal.
          continue;
        }
        else
        {
          // Failed to acquire semaphore.
          abort();
        }
      }
    }
  };
} // namespace verona::rt::pal
#  else
#    error "No semaphore implementation available"
#  endif
namespace verona::rt::pal
{
  /**
   * Handles thread sleeping.
   */
  class SleepHandle
  {
    SemaphoreImpl sem;

#  ifndef NDEBUG
    std::atomic<bool> sleeper{false};
    std::atomic<bool> waker{false};
#  endif

  public:
    /**
     * Called to sleep until a matching call to wake is made.
     *
     * There are not allowed to be two parallel calls to sleep.
     */
    void sleep()
    {
#  ifndef NDEBUG
      assert(!sleeper);
      sleeper = true;
#  endif
      sem.acquire();
#  ifndef NDEBUG
      waker = false;
      sleeper = false;
#  endif
    }

    /**
     * Used to wake a thread from sleep.
     *
     * The number of calls to wake may be at most one more than the number of
     * calls to sleep.
     */
    void wake()
    {
      assert(!waker);
#  ifndef NDEBUG
      waker = true;
#  endif
      sem.release();
    }
  };
} // namespace verona::rt::pal
#endif
