// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <random>
#include <test/harness.h>
#include <test/opt.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

struct A : public VCown<A>
{};

struct M : public VBehaviour<M>
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

struct Dummy : public VCown<Dummy>
{};

struct RemoveExternalEvent : public VBehaviour<RemoveExternalEvent>
{
  Cown* a;

  RemoveExternalEvent(Cown* a) : a(a) {}

  void f()
  {
    Systematic::cout() << "Remove external event source" << std::endl;
    Scheduler::remove_external_event_source();
    Cown::release(ThreadAlloc::get(), a);
  }
};

struct StartExternalThread : public VBehaviour<StartExternalThread>
{
  size_t pauses;

  StartExternalThread(size_t pauses) : pauses(pauses) {}

  void f()
  {
    auto a = new A;
    Scheduler::add_external_event_source();
    auto pauses_ = pauses;
    auto t = std::thread([pauses_, a]() mutable {
      Systematic::cout() << "Started external thread" << Systematic::endl;
      std::mt19937 rng;
      rng.seed(1);
      std::uniform_int_distribution<> dist(1, 1000);
      for (size_t i = 1; i <= pauses_; i++)
      {
        auto pause_time = std::chrono::milliseconds(dist(rng));
        std::this_thread::sleep_for(pause_time);
        Systematic::cout() << "Scheduling Message" << Systematic::endl;
        Cown::schedule<M>(a, a, i, i == pauses_);
      }

      auto e = new Dummy;
      Cown::schedule<RemoveExternalEvent>(e, e);

      Systematic::cout() << "External thread exiting" << Systematic::endl;
    });
    t.detach();
  }
};

void test_runtime_pause(size_t pauses)
{
  auto e = new Dummy;
  Cown::schedule<StartExternalThread>(e, pauses);
  Cown::release(ThreadAlloc::get(), e);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  size_t pauses = harness.opt.is<size_t>("--pauses", 3);

  harness.run(test_runtime_pause, pauses);
  return 0;
}
