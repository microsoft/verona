// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 *  This benchmark is for testing performance of the scheduling code.
 *
 * There are n cowns, each executing m writes to a large statically allocated
 * array of memory.  Each cown performs c behaviours.
 */

#include "test/log.h"
#include "test/opt.h"
#include "test/xoroshiro.h"
#include "verona.h"

#include <chrono>
#include <test/harness.h>


namespace sn = snmalloc;
namespace rt = verona::rt;

// Memory to use for workload
size_t * global_array;
size_t global_array_size;

// Number of writes on each iteration
size_t writes;

struct LoopCown : public VCown<LoopCown>
{
  size_t count;

  LoopCown(size_t count) : count(count) {}

  void go()
  {
    if (count > 0)
    {
      count --;
      schedule_lambda(this, [this]() { work(); go(); });
    }
    else
    {
      Cown::release(ThreadAlloc::get(), this);
    }
  }

  void work()
  {
  }
};

int main(int argc, char** argv)
{
  for (int i = 0; i < argc; i++)
  {
      printf(" %s", argv[i]);
  }
  printf("\n");
  opt::Opt opt(argc, argv);

//  auto& alloc = sn::ThreadAlloc::get();

  const auto cores = opt.is<size_t>("--cores", 4);
  const auto cowns = (size_t)1 << opt.is<size_t>("--cowns", 8);
  global_array_size  = (size_t)1 << opt.is<size_t>("--size", 22);
  global_array = new size_t[global_array_size];
  const auto loops = opt.is<size_t>("--loops", 100);
  writes = opt.is<size_t>("--writes", 100);

  auto& sched = rt::Scheduler::get();
  sched.set_fair(true);
  for (int l = 0; l < 20; l++)
  {
    sched.init(cores);

    for (size_t i = 0; i < cowns; i++)
    {
        auto c = new LoopCown(loops);
        c->go();
    }

    auto start = sn::Aal::tick();
    sched.run();
    auto end = sn::Aal::tick();
    std::cout << "Time:" << (end - start) / (cowns * loops) << std::endl;
  }
  delete [] global_array;
  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}
