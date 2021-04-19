// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <snmalloc.h>

#if defined(__linux__)
#  include <sched.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <unistd.h>
#elif defined(_WIN32)
#  include <processtopologyapi.h>
#elif defined(__FreeBSD__)
#  include <sys/cpuset.h>
#  ifdef _KERNEL
#    include <sys/pcpu.h>
#    include <sys/proc.h>
#  else
#    include <pthread.h>
#    include <pthread_np.h>
#  endif
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <mach/thread_policy.h>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <vector>

namespace verona::rt
{
  using namespace snmalloc;
  class Topology
  {
  private:
    struct CPU
    {
      size_t numa_node;
      size_t package;
      size_t group;
      size_t id;
      bool hyperthread;

      size_t get()
      {
#if defined(_WIN32)
        return (group << 6) + id;
#else
        return id;
#endif
      }

      bool operator<(const CPU& that) const
      {
        // Prefer physical cores.
        if (hyperthread != that.hyperthread)
          return !hyperthread;

        // Sort by numa node.
        if (numa_node < that.numa_node)
          return true;

        if (numa_node > that.numa_node)
          return false;

        // Sort by package within a numa node.
        if (package < that.package)
          return true;

        if (package > that.package)
          return false;

        // Sort by group.
        if (group < that.group)
          return true;

        if (group > that.group)
          return false;

        // Sort by id.
        return id < that.id;
      }
    };

    std::vector<CPU>* cpus = nullptr;

#if defined(CPU_COUNT) && defined(CPU_ISSET)
    template<typename CPUSet>
    void get_cpuset(std::function<void(CPUSet&)> getaffinity)
    {
      CPUSet all_cpus;

      getaffinity(all_cpus);
      size_t count = (size_t)CPU_COUNT(&all_cpus);
      uint32_t index = 0;
      uint32_t found = 0;

      // TODO: CPU topology detection for Linux
      while (found < count)
      {
        if (CPU_ISSET(index, &all_cpus))
        {
          cpus->push_back(CPU{0, 0, 0, index, false});
          found++;
        }

        index++;
      }
    }
#endif

  public:
    ~Topology()
    {
      release();
    }

    void acquire()
    {
      delete cpus;
      cpus = new std::vector<CPU>;

#if defined(_WIN32)
      size_t numa_count;
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX numa =
        get_info(RelationNumaNode, numa_count);

      size_t package_count;
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX package =
        get_info(RelationProcessorPackage, package_count);

      size_t core_count;
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX core =
        get_info(RelationProcessorCore, core_count);

      auto p = core;

      for (size_t i = 0; i < core_count; i++)
      {
        for (size_t j = 0; j < p->Processor.GroupCount; j++)
        {
          size_t group = p->Processor.GroupMask[j].Group;
          bool hyperthread = false;

          for (size_t id = 0; id < 64; id++)
          {
            size_t idmask = (size_t)1 << id;

            if (idmask & p->Processor.GroupMask[j].Mask)
            {
              cpus->push_back(
                CPU{get_numa_node(group, id, numa, numa_count),
                    get_package(group, id, package, package_count),
                    group,
                    id,
                    hyperthread});

              hyperthread = true;
            }
          }
        }

        p = next_info(p);
      }

      release_info(core);
      release_info(package);
      release_info(numa);
#elif defined(__linux__)
      get_cpuset<cpu_set_t>([](cpu_set_t& all_cpus) {
        sched_getaffinity(0, sizeof(cpu_set_t), &all_cpus);
      });
#elif defined(FreeBSD_KERNEL)
      get_cpuset<cpuset_t>(
        [](cpuset_t& all_cpus) { CPU_COPY(cpuset_root, &all_cpus); });
#elif defined(__FreeBSD__)
      get_cpuset<cpuset_t>([](cpuset_t& all_cpus) {
        pthread_getaffinity_np(pthread_self(), sizeof(cpuset_t), &all_cpus);
      });
#elif defined(__APPLE__)
      // This code only runs once per process, when initializing the runtime.
      // There isn't any point caching sysctl's MIB, so we simply use
      // sysctlbyname instead.
      unsigned int core_count;
      size_t len = sizeof(core_count);
      int ret = sysctlbyname("hw.logicalcpu", &core_count, &len, NULL, 0);

      // If sysctlbyname failed we can leave cpus empty and everything will
      // work, we just won't get CPU affinity.
      if (ret >= 0)
      {
        cpus->reserve(core_count);
        for (uint32_t index = 0; index < core_count; index++)
        {
          cpus->push_back(CPU{0, 0, 0, index, false});
        }
      }
#else
#  error Missing CPU enumeration for your OS.
#endif

      std::sort(cpus->begin(), cpus->end());
    }

    void release()
    {
      delete cpus;
      cpus = nullptr;
    }

    size_t get(size_t index)
    {
      if ((cpus == nullptr) || (cpus->size() == 0))
        abort();

      index = index % cpus->size();
      return cpus->at(index).get();
    }

    size_t size()
    {
      if ((cpus == nullptr) || (cpus->size() == 0))
        abort();

      return cpus->size();
    }

  private:
#ifdef _WIN32
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
    get_info(LOGICAL_PROCESSOR_RELATIONSHIP relation, size_t& count)
    {
      DWORD len = 0;
      count = 0;

      if (GetLogicalProcessorInformationEx(relation, nullptr, &len))
        return nullptr;

      if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return nullptr;

      char* buffer = new char[len];
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer;

      if (!GetLogicalProcessorInformationEx(relation, info, &len))
      {
        delete[] buffer;
        return nullptr;
      }

      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = info;

      while (len > 0)
      {
        len -= p->Size;
        p = next_info(p);
        count++;
      }

      return info;
    }

    size_t get_numa_node(
      size_t group,
      size_t id,
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info,
      size_t count)
    {
      size_t idmask = (size_t)1 << id;

      for (size_t i = 0; i < count; i++)
      {
        if (
          (group == info->NumaNode.GroupMask.Group) &&
          (idmask & info->NumaNode.GroupMask.Mask))
        {
          return info->NumaNode.NodeNumber;
        }

        info = next_info(info);
      }

      return 0;
    }

    size_t get_package(
      size_t group,
      size_t id,
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info,
      size_t count)
    {
      size_t idmask = (size_t)1 << id;

      for (size_t i = 0; i < count; i++)
      {
        for (size_t j = 0; j < info->Processor.GroupCount; j++)
        {
          if (
            (group == info->Processor.GroupMask[j].Group) &&
            (idmask & info->Processor.GroupMask[j].Mask))
          {
            return i;
          }
        }

        info = next_info(info);
      }

      return 0;
    }

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
    next_info(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info)
    {
      return (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(
        (char*)info + info->Size);
    }

    void release_info(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info)
    {
      if (info != nullptr)
        delete[]((char*)info);
    }
#endif
  };

  namespace cpu
  {
    inline void set_affinity(size_t affinity)
    {
      if (affinity == (size_t)-1)
        return;

#if defined(_WIN32)
      GROUP_AFFINITY g;
      g.Mask = (uint64_t)1 << (affinity % 64);
      g.Group = (WORD)(affinity / 64);

      SetThreadGroupAffinity(GetCurrentThread(), &g, NULL);
#elif defined(__linux__)
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(affinity, &set);

      sched_setaffinity(0, sizeof(cpu_set_t), &set);
#elif defined(__FreeBSD__)
      cpuset_t set;
      CPU_ZERO(&set);
      CPU_SET(affinity, &set);

#  ifdef _KERNEL
      cpuset_setthread(curthread->td_tid, &set);
#  else
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#  endif
#elif defined(__APPLE__)
      thread_affinity_policy_data_t policy = {static_cast<integer_t>(affinity)};

      thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        1);
#else
#  error Missing CPU affinity support for your OS.
#endif
    }
  } // namespace cpu
} // namespace verona::rt
