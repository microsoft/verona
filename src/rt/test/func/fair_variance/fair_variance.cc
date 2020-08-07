// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ctime>
#include <test/harness.h>
#include <test/opt.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

static constexpr int start_count = 1'00'000;
struct A : public VCown<A>
{
  int id;
  int count = start_count;
  clock_t begin;

  A(int id_) : id{id_} {}
};

int constexpr n_cowns = 6;
double elapsed_secs[n_cowns];

struct Loop : public VBehaviour<Loop>
{
  A* a;
  Loop(A* a) : a(a) {}

  void f()
  {
    auto& count = a->count;
    auto id = a->id;

    if (count == start_count)
    {
      a->begin = clock();
    }

    if (count == 0)
    {
      clock_t end = clock();
      double elapsed_second = double(end - a->begin) / CLOCKS_PER_SEC;
      elapsed_secs[id] = elapsed_second;
      // printf("%d: %f\n", a->id, elapsed_second);
      return;
    }

    count--;
    Cown::schedule<Loop>(a, a);
  }
};

struct B : public VCown<A>
{};

struct Spawn : public VBehaviour<Spawn>
{
  void f()
  {
    auto* alloc = ThreadAlloc::get();
    (void)alloc;
    for (int i = 0; i < n_cowns; ++i)
    {
      auto a = new A(i);
      Cown::schedule<Loop>(a, a);
      Cown::release(alloc, a);
    }
  }
};

void assert_variance()
{
  using namespace std;
  auto result = minmax_element(elapsed_secs, elapsed_secs + n_cowns);
  auto min = *result.first;
  auto max = *result.second;
  check(min != 0 && max != 0);
  // printf("%f\n", (max - min)/max);
  // variance should be less than 15%
  if ((max - min) / max > 0.15)
  {
    printf("(max - min) / max = %f\n", (max - min) / max);
    for (int i = 0; i < n_cowns; i++)
    {
      printf("cown[%d] took %f\n", i, elapsed_secs[i]);
    }
    check(!"variance too large");
  }
  UNUSED(min);
  UNUSED(max);
}

int main()
{
#ifdef USE_SYSTEMATIC_TESTING
  std::cout << "This test does not make sense to run systematically."
            << std::endl;
#else
  size_t cores = 2;
  Scheduler& sched = Scheduler::get();
  sched.init(cores);
  sched.set_fair(true);

  auto* alloc = ThreadAlloc::get();
  (void)alloc;

  auto b = new B;
  Cown::schedule<Spawn>(b);

  Cown::release(alloc, b);
  sched.run();
  snmalloc::current_alloc_pool()->debug_check_empty();
  assert_variance();

  puts("done");
#endif
  return 0;
}
