// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include <cassert>
#ifdef _WIN32
#  include <windows.h>
#elif defined(VERONA_EXTERNAL_THREADING)
#  include <verona_external_threading.h>
#else
#  include <functional>
#  include <mutex>
#  include <unistd.h>
#  ifdef FreeBSD_KERNEL
extern "C"
{
#    include <sys/smp.h>
}
#  else
#    include <sys/mman.h>
#  endif
#  ifdef __linux__
#    include <linux/membarrier.h>
#    include <sys/syscall.h>
#    ifndef MEMBARRIER_CMD_PRIVATE_EXPEDITED
#      define MEMBARRIER_CMD_PRIVATE_EXPEDITED 0
#    endif
#    ifndef MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED
#      define MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED 0
#    endif
#  endif
#endif

#include <snmalloc.h>

namespace verona::rt
{
  class Barrier
  {
  private:
#if !defined(_WIN32) && !defined(VERONA_EXTERNAL_THREADING)
#  ifdef __linux__
    /**
     * Linux version of `FlushProcessWriteBuffers`, uses membarrier.
     */
    static void FlushProcessWriteBuffers()
    {
      // Helper to issue the system call.
      static auto membarrier = [](int cmd, int flags) {
        return (int)syscall(__NR_membarrier, cmd, flags);
      };

      static bool broken_membarrier = false;
      // If `USE_MEMBARRIER_EXPEDITED` is defined to one, then the first time
      // we enter this function try to register the command that we want
      // otherwise just use a shared barrier.
      static int cmd = []() -> int {
        int r = membarrier(MEMBARRIER_CMD_QUERY, 0);

        if (r == -1)
        {
          if (membarrier(MEMBARRIER_CMD_SHARED, 0) != -1)
          {
            return MEMBARRIER_CMD_SHARED;
          }
          broken_membarrier = true;
          return -1;
        }

        if (r & MEMBARRIER_CMD_PRIVATE_EXPEDITED)
        {
          membarrier(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0);
          return MEMBARRIER_CMD_PRIVATE_EXPEDITED;
        }

        if (r & MEMBARRIER_CMD_SHARED)
        {
          return MEMBARRIER_CMD_SHARED;
        }

        broken_membarrier = true;
        return -1;
      }();

      if (broken_membarrier)
      {
        FlushProcessWriteBuffersPortable();
      }
      else
      {
        membarrier(cmd, 0);
      }
    }
#    define FlushProcessWriteBuffers FlushProcessWriteBuffersPortable
#  endif
#  ifdef FreeBSD_KERNEL
    static void FlushProcessWriteBuffers()
    {
      smp_rendezvous(nullptr, nullptr, nullptr, nullptr);
    }
#  else
    /**
     * Portable implementation of `FlushProcessWriteBuffers` for non-Windows
     * platforms.  This ensures that all other threads have seen writes by us,
     * even if they don't issue any barriers.
     *
     * This implementation is based on the version in CoreCLR, which updates
     * page permissions to force the OS to do an IPI to perform page
     * shootdowns.
     */
    static void FlushProcessWriteBuffers()
    {
      static std::mutex m;
      std::lock_guard<std::mutex> g(m);
      static std::atomic<int>* page;
      static std::once_flag f;
      static size_t page_size;
      static auto die_if = [](bool cond, const char* msg) {
        if (cond)
        {
          perror(msg);
          abort();
        }
      };

      std::call_once(f, [&]() {
        page_size = (size_t)sysconf(_SC_PAGESIZE);
        page = static_cast<std::atomic<int>*>(mmap(
          nullptr,
          page_size,
          PROT_READ | PROT_WRITE,
          MAP_ANON | MAP_PRIVATE,
          -1,
          0));
        die_if(page == MAP_FAILED, "Unable to map one page of memory.");
        die_if(mlock(page, page_size) != 0, "Unable to lock memory.");
      });
      int status = mprotect(page, page_size, PROT_READ | PROT_WRITE);
      page[0]++;
      status |= mprotect(page, page_size, PROT_NONE);
      die_if(status != 0, "Failed in mprotect call to trigger memory barrier.");
    }
#  endif
#  ifdef FlushProcessWriteBuffers
#    undef FlushProcessWriteBuffers
#  endif
#endif

  public:
    static inline void memory()
    {
      FlushProcessWriteBuffers();
    }

    static inline void compiler()
    {
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }
  };
} // namespace verona::rt
