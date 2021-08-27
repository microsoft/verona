#pragma once

namespace verona::rt
{
  class Topology
  {
  public:
    Topology();
    Topology(const Topology&) = delete;
    Topology(Topology&&) = delete;

    ~Topology(){};

    void acquire() {}
    void release() {}
    size_t get(size_t index)
    {
      return index;
    }
  };

  namespace cpu
  {
    inline void set_affinity(size_t) {}
  }

  class InternalThread
  {
  public:
    template<typename F, typename... Args>
    InternalThread(F&&, Args&&...)
    {}

    InternalThread() = delete;
    InternalThread(const InternalThread&) = delete;
    InternalThread(InternalThread&&) = delete;

    ~InternalThread(){};

    void join() {}
  };

  inline void FlushProcessWriteBuffers() {}
}
