// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <random>
#include <test/opt.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

struct A : public VCown<A>
{};

struct M : public VAction<M>
{
  Cown* a;

  size_t id;
  bool last = false;

  M(Cown* a, size_t i, bool l = false) : a(a), id(i), last(l) {}

  void f()
  {
    Systematic::cout() << "running message " << id << std::endl;

    if (last)
      Cown::release(ThreadAlloc::get(), a);
  }
};

void test_runtime_pause(size_t cores, size_t pauses)
{
  Scheduler& sched = Scheduler::get();
  sched.init(cores);
  Scheduler::set_allow_teardown(false);

  auto a = new A;

  auto thr = std::thread([pauses, &a]() mutable {
    std::mt19937 rng;
    rng.seed(1);
    std::uniform_int_distribution<> dist(1, 1000);
    for (size_t i = 1; i <= pauses; i++)
    {
      auto pause_time = std::chrono::milliseconds(dist(rng));
      std::this_thread::sleep_for(pause_time);
      Cown::schedule<M>(a, a, i, i == pauses);
    }

    auto pause_time = std::chrono::nanoseconds(dist(rng));
    std::this_thread::sleep_for(pause_time);

    Scheduler::set_allow_teardown(true);
  });

  sched.run();
  thr.join();
  snmalloc::current_alloc_pool()->debug_check_empty();
}

int main(int argc, char** argv)
{
#ifdef USE_SYSTEMATIC_TESTING
  std::cout << "Testing external concurrency, so cannot use systematic testing."
            << std::endl;
  UNUSED(argc);
  UNUSED(argv);
#else
  opt::Opt opt(argc, argv);
  size_t cores = opt.is<size_t>("--cores", 4);
  size_t pauses = opt.is<size_t>("--pauses", 3);

  for (size_t i = 0; i < 100; i++)
  {
    std::cout << "Repeat: " << i << std::endl;
    test_runtime_pause(cores, pauses);
  }
#endif
  return 0;
}
