// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This constructs a platform specific semaphore.
 */
#if __has_include(<semaphore>)
#  include <semaphore>
#endif
#if defined(__cpp_lib_semaphore)
namespace verona::rt::pal
{
  class Semaphore
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
#elif defined(__APPLE__)
#  include <dispatch/dispatch.h>
namespace verona::rt::pal
{
  class Semaphore
  {
    dispatch_semaphore_t semaphore_;

  public:
    Semaphore()
    {
      semaphore_ = dispatch_semaphore_create(0);
    }

    ~Semaphore()
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
#elif defined(WIN32)
#  include <windows.h>
namespace verona::rt::pal
{
  class Semaphore
  {
    HANDLE semaphore_;

  public:
    Semaphore()
    {
      semaphore_ = CreateSemaphore(NULL, 0, LONG_MAX, NULL);
    }

    ~Semaphore()
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
#elif __has_include(<semaphore.h>)
// Use Posix semaphores
#  include <semaphore.h>
namespace verona::rt::pal
{
  class Semaphore
  {
    sem_t semaphore_;

  public:
    Semaphore()
    {
      auto err = sem_init(&semaphore_, 0, 0);
      if (err != 0)
      {
        // Failed to initialize semaphore.
        abort();
      }
    }

    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

    ~Semaphore()
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
#else
#  error "No semaphore implementation available"
#endif
