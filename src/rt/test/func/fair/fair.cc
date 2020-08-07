// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ctime>
#include <test/harness.h>

static constexpr int start_count = 100;
struct A : public VCown<A>
{
  int id;
  int count = start_count;
  clock_t begin;

  A(int id_) : id{id_} {}
};

struct Loop : public VBehaviour<Loop>
{
  A* a;
  Loop(A* a) : a(a) {}

  void f()
  {
    auto& count = a->count;

    if (count == start_count)
    {
      a->begin = clock();
    }

    if (count == 0)
    {
      clock_t end = clock();
      double elapsed_secs = double(end - a->begin) / CLOCKS_PER_SEC;
      (void)elapsed_secs;
      // printf("%d: %f\n", a->id, elapsed_secs);
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
    for (int i = 0; i < 6; ++i)
    {
      auto a = new A(i);
      Cown::schedule<Loop>(a, a);
      Cown::release(alloc, a);
    }
  }
};

void basic_test()
{
  auto* alloc = ThreadAlloc::get();

  auto b = new B;
  Cown::schedule<Spawn>(b);

  Cown::release(alloc, b);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(basic_test);
  return 0;
}
