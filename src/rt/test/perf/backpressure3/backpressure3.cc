// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This test involves many senders and a single receiver. Whenever the receiver
 * receives a message from the sender it will randomly select a sender with
 * which to run a multi-message. This is intended to test a scenario where a
 * muted cowns are frequently required by an overloaded cown to make progress.
 *
 * A correct implementation of backpressure must ensure that the receivers make
 * progress despite requiring their muted senders to do so.
 */

#include "test/log.h"
#include "test/opt.h"
#include "test/xoroshiro.h"
#include "verona.h"

#include <chrono>

using namespace verona::rt;
using timer = std::chrono::high_resolution_clock;

struct Sender;

struct Receiver : public VCown<Receiver>
{
  std::vector<Sender*>& senders;
  xoroshiro::p128r32 rng;
  size_t msgs = 0;
  timer::time_point prev = timer::now();

  Receiver(std::vector<Sender*>& senders_, size_t seed)
  : senders(senders_), rng(seed)
  {}

  void trace(ObjectStack& st) const
  {
    for (auto* s : senders)
      st.push((Object*)s);
  }
};

struct Receive : public VBehaviour<Receive>
{
  Receiver* r;
  Sender* s;

  Receive(Receiver* r_, Sender* s_ = nullptr) : r(r_), s(s_) {}

  void f()
  {
    auto& alloc = ThreadAlloc::get();
    if (s == nullptr)
    {
      s = r->senders[r->rng.next() % r->senders.size()];
      auto** cowns = (Cown**)alloc.alloc<2 * sizeof(Cown*)>();
      cowns[0] = (Cown*)r;
      cowns[1] = (Cown*)s;
      Cown::schedule<Receive>(2, cowns, r, s);
      alloc.dealloc<2 * sizeof(Cown*)>(cowns);
    }
    else
    {
      r->msgs++;

      const auto now = timer::now();
      const auto t =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - r->prev);
      if (r->msgs < 100'000)
        return;

      logger::cout() << r << " received " << r->msgs << " messages in "
                     << t.count() << "ms" << std::endl;
      r->prev = now;
      r->msgs = 0;
    }
  }
};

struct Sender : public VCown<Sender>
{
  timer::time_point start = timer::now();
  timer::duration duration;
  Receiver* receiver;

  Sender(timer::duration duration_, Receiver* receiver_)
  : duration(duration_), receiver(receiver_)
  {}

  void trace(ObjectStack& st) const
  {
    if (receiver != nullptr)
      st.push(receiver);
  }
};

struct Send : public VBehaviour<Send>
{
  Sender* s;

  Send(Sender* s_) : s(s_) {}

  void f()
  {
    Cown::schedule<Receive>(s->receiver, s->receiver);

    if ((timer::now() - s->start) < s->duration)
      Cown::schedule<Send>(s, s);
    else
    {
      // Break cycle between sender and receiver.
      Cown::release(ThreadAlloc::get(), s->receiver);
      s->receiver = nullptr;
    }
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  const auto seed = opt.is<size_t>("--seed", 5489);
  const auto cores = opt.is<size_t>("--cores", 4);
  const auto senders = opt.is<size_t>("--senders", 100);
  const auto duration =
    std::chrono::milliseconds(opt.is<size_t>("--duration", 10'000));

  logger::cout() << "cores: " << cores << ", senders: " << senders
                 << ", duration: " << duration.count() << "ms" << std::endl;

#ifdef USE_SYSTEMATIC_TESTING
  Logging::enable_logging();
  Systematic::set_seed(seed);
#endif
  Scheduler::set_detect_leaks(true);
  auto& sched = Scheduler::get();
  sched.set_fair(true);
  sched.init(cores);

  auto& alloc = ThreadAlloc::get();

  static std::vector<Sender*> sender_set;
  auto* receiver = new (alloc) Receiver(sender_set, seed);

  for (size_t s = 0; s < senders; s++)
  {
    Cown::acquire(receiver);
    sender_set.push_back(new (alloc) Sender(duration, receiver));
  }

  for (auto* s : sender_set)
    Cown::schedule<Send, NoTransfer>(s, s);
  Cown::release(alloc, receiver);

  sched.run();

  return 0;
}
