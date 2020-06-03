// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This test involves many small sets of `Sender` cowns repeatedly selecting a
 * small random subset of `Receiver` cowns and sending them a message.
 **/

#include "test/log.h"
#include "test/opt.h"
#include "test/xoroshiro.h"
#include "verona.h"

#include <chrono>

using namespace verona::rt;
using timer = std::chrono::high_resolution_clock;

template<typename RNG>
struct RNGWrapper
{
  RNG rng;
  using result_type = typeof(rng.next());

  RNGWrapper(size_t seed1, size_t seed2) : rng(seed1, seed2) {}

  result_type min() const
  {
    return std::numeric_limits<result_type>::min();
  }

  result_type max() const
  {
    return std::numeric_limits<result_type>::max();
  }

  result_type operator()()
  {
    return rng.next();
  }
};

struct Receiver : public VCown<Receiver>
{
  size_t msgs = 0;
  timer::time_point prev = timer::now();
};

struct Receive : public VAction<Receive>
{
  Receiver** receivers;
  size_t receiver_count;

  Receive(Receiver** receivers_, size_t receiver_count_)
  : receivers(receivers_), receiver_count(receiver_count_)
  {}

  ~Receive()
  {
    ThreadAlloc::get()->dealloc(receivers, receiver_count * sizeof(Receiver*));
  }

  void trace(ObjectStack& st) const
  {
    for (size_t i = 0; i < receiver_count; i++)
      st.push(receivers[i]);
  }

  void f()
  {
    for (size_t i = 0; i < receiver_count; i++)
    {
      auto& r = *receivers[i];
      r.msgs++;
      const auto now = timer::now();
      const auto t =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - r.prev);
      if (t < std::chrono::milliseconds(1000))
        return;

      logger::cout() << &r << " received " << r.msgs << " messages"
                     << std::endl;
      r.prev = now;
      r.msgs = 0;
    }
  }
};

struct Sender : public VCown<Sender>
{
  timer::time_point start = timer::now();
  timer::duration duration;
  std::vector<Receiver*>& receivers;
  RNGWrapper<xoroshiro::p128r32> rng;

  Sender(
    timer::duration duration_,
    std::vector<Receiver*>& receivers_,
    size_t seed1,
    size_t seed2)
  : duration(duration_), receivers(receivers_), rng(seed1, seed2)
  {}

  void trace(ObjectStack& st) const
  {
    for (auto* r : receivers)
      st.push(r);
  }
};

struct Send : public VAction<Send>
{
  Sender* s;

  Send(Sender* s_) : s(s_) {}

  void f()
  {
    static constexpr size_t max_receivers = 3;
    const size_t receiver_count = std::uniform_int_distribution<size_t>(
      1, (std::min)(max_receivers, s->receivers.size()))(s->rng);

    auto** receivers =
      (Receiver**)ThreadAlloc::get()->alloc(receiver_count * sizeof(Receiver*));

    auto index_rng =
      std::uniform_int_distribution<size_t>(0, s->receivers.size() - 1);
    for (size_t i = 0; i < receiver_count;)
    {
      receivers[i] = s->receivers[index_rng(s->rng)];
      if (std::find(receivers, &receivers[i], receivers[i]) == &receivers[i])
        i++;
    }

    Cown::schedule<Receive>(
      receiver_count, (Cown**)receivers, receivers, receiver_count);

    if ((timer::now() - s->start) < s->duration)
      Cown::schedule<Send>(s, s);
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  auto seed = opt.is<size_t>("--seed", 5489);
  auto cores = opt.is<size_t>("--cores", 4);
  auto senders = opt.is<size_t>("--senders", 100);
  auto receivers = opt.is<size_t>("--receivers", 10);
  auto duration = opt.is<size_t>("--duration", 10'000);
  logger::cout() << "cores: " << cores << ", senders: " << senders
                 << ", receivers: " << receivers << ", duration: " << duration
                 << "ms" << std::endl;

  auto& sched = Scheduler::get();
  Scheduler::set_detect_leaks(true);
#ifdef USE_SYSTEMATIC_TESTING
  Systematic::enable_logging();
  sched.set_seed(seed);
#endif
  sched.set_fair(true);
  sched.init(cores);

  auto* alloc = ThreadAlloc::get();

  static std::vector<Receiver*> receiver_set;
  for (size_t i = 0; i < receivers; i++)
    receiver_set.push_back(new (alloc) Receiver);

  xoroshiro::p128r32 rng(seed);
  for (size_t i = 0; i < senders; i++)
  {
    for (auto* r : receiver_set)
      Cown::acquire(r);

    auto* s = new (alloc) Sender(
      std::chrono::milliseconds(duration), receiver_set, seed, rng.next());
    Cown::schedule<Send, YesTransfer>(s, s);
  }

  for (auto* r : receiver_set)
    Cown::release(alloc, r);

  sched.run();
}
