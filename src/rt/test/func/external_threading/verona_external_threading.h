// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::rt
{
  /**
   * Helper class to help select CPU cores for thread affinity.
   */
  class Topology
  {
  public:
    Topology(){};
    Topology(const Topology&) = delete;
    Topology(Topology&&) = delete;

    ~Topology(){};

    /**
     * Signals the the start of the Topology usage.
     * No calls should be made to Topology::get before calling this method.
     */
    static void init(Topology*) noexcept {}

    /**
     * Assigns a CPU ID to a scheduler thread specified as the argument.
     */
    size_t get(size_t index)
    {
      return index;
    }
  };

  namespace cpu
  {
    /**
     * Moves the current thread onto the CPU with the given ID.
     * This ID is one returned by Topology::get() earlier.
     */
    inline void set_affinity(size_t) {}
  }

  class PlatformThread
  {
  public:
    /**
     * Creates a new thread and executes the function with the given arguments
     * on it.
     */
    template<typename F, typename... Args>
    PlatformThread(F&&, Args&&...)
    {}

    PlatformThread() = delete;
    PlatformThread(const PlatformThread&) = delete;
    PlatformThread(PlatformThread&&) = delete;

    /**
     * Destructor to be called after the thread has finished executing (and was
     * joined).
     */
    ~PlatformThread(){};

    /**
     * Wait until the thread has finished executing the function it was
     * initialized with.
     */
    void join() {}
  };

  /**
   * Flushes the write queue of each processor that is running a thread of the
   * current process.
   */
  inline void FlushProcessWriteBuffers() {}
}
