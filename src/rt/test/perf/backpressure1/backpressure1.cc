// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This tests a simple scenario for backpressure where many individual `Sender`
 * cowns send messages to a single set of `Receiver` cowns. The `Recevier` cowns
 * may be placed behind a chain of `Proxy` cowns to test backpressure
 * propagation.
 *
 * Without backpressure, the receivers would have their queues grow at a much
 * higher rate than they could process the messages. The muted proxies may also
 * experience similar queue growth if the backpressure is not corretly
 * propagated from the receiver set.
 */

#include "test/log.h"
#include "test/opt.h"
#include "verona.h"

#include <chrono>

using namespace verona::rt;
using timer = std::chrono::high_resolution_clock;

struct Receiver;
struct Proxy;
static std::vector<Receiver*> receiver_set;
static std::vector<Proxy*> proxy_chain;

struct Receiver : public VCown<Receiver>
{
  static constexpr size_t report_count = 1'000'000;
  size_t msgs = 0;
  timer::time_point prev = timer::now();
};

struct Receive : public VBehaviour<Receive>
{
  void f()
  {
    auto& r = *receiver_set[0];
    r.msgs++;
    if ((r.msgs % Receiver::report_count) != 0)
      return;

    const auto now = timer::now();
    const auto t =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - r.prev);
    logger::cout() << Receiver::report_count << " messages received in "
                   << t.count() << "ms" << std::endl;
    r.prev = now;
  }
};

struct Proxy : public VCown<Proxy>
{
  size_t index;

  Proxy(size_t index_) : index(index_) {}

  void trace(ObjectStack& st) const
  {
    if (this != proxy_chain.back())
    {
      st.push(proxy_chain[index + 1]);
      return;
    }

    for (auto* r : receiver_set)
      st.push(r);
  }
};

struct Forward : public VBehaviour<Forward>
{
  Proxy* proxy;

  Forward(Proxy* proxy_) : proxy(proxy_) {}

  void f()
  {
    if (proxy != proxy_chain.back())
    {
      auto* next = proxy_chain[proxy->index + 1];
      Cown::schedule<Forward>(next, next);
      return;
    }

    Cown::schedule<Receive>(receiver_set.size(), (Cown**)receiver_set.data());
  }
};

struct Sender : public VCown<Sender>
{
  using clk = std::chrono::steady_clock;

  clk::time_point start = clk::now();
  std::chrono::milliseconds duration;

  Sender(std::chrono::milliseconds duration_) : duration(duration_) {}

  void trace(ObjectStack& st) const
  {
    if (proxy_chain.size() > 0)
    {
      st.push(proxy_chain[0]);
      return;
    }

    for (auto* r : receiver_set)
      st.push(r);
  }
};

struct Send : public VBehaviour<Send>
{
  Sender* s;

  Send(Sender* s_) : s(s_) {}

  void f()
  {
    if (proxy_chain.size() > 0)
      Cown::schedule<Forward>(proxy_chain[0], proxy_chain[0]);
    else
      Cown::schedule<Receive>(receiver_set.size(), (Cown**)receiver_set.data());

    if ((Sender::clk::now() - s->start) < s->duration)
      Cown::schedule<Send>(s, s);
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  auto seed = opt.is<size_t>("--seed", 5489);
  auto cores = opt.is<size_t>("--cores", 4);
  auto senders = opt.is<size_t>("--senders", 100);
  auto receivers = opt.is<size_t>("--receivers", 1);
  auto proxies = opt.is<size_t>("--proxies", 0);
  auto duration = opt.is<size_t>("--duration", 10'000);
  logger::cout() << "cores: " << cores << ", senders: " << senders
                 << ", receivers: " << receivers << ", duration: " << duration
                 << "ms" << std::endl;

#ifdef USE_SYSTEMATIC_TESTING
  Systematic::enable_logging();
  Systematic::set_seed(seed);
#else
  UNUSED(seed);
#endif
  Scheduler::set_detect_leaks(true);
  auto& sched = Scheduler::get();
  sched.set_fair(true);
  sched.init(cores);

  Alloc& alloc = ThreadAlloc::get();

  for (size_t r = 0; r < receivers; r++)
    receiver_set.push_back(new (alloc) Receiver);

  for (size_t p = 0; p < proxies; p++)
    proxy_chain.push_back(new (alloc) Proxy(p));

  auto* e = new EmptyCown;
  schedule_lambda(e, [] {
    Systematic::cout() << "Add external event source" << std::endl;
    Scheduler::add_external_event_source();
  });

  auto thr = std::thread([=, &alloc] {
    for (size_t i = 0; i < senders; i++)
    {
      if (proxy_chain.size() > 0)
      {
        Cown::acquire(proxy_chain[0]);
      }
      else
      {
        for (auto* r : receiver_set)
          Cown::acquire(r);
      }

      auto* s = new (alloc) Sender(std::chrono::milliseconds(duration));
      Cown::schedule<Send, YesTransfer>(s, s);
    }

    if (proxy_chain.size() > 0)
    {
      Cown::release(alloc, proxy_chain[0]);
    }
    else
    {
      for (auto* r : receiver_set)
        Cown::release(alloc, r);
    }

    schedule_lambda(e, [e] {
      Systematic::cout() << "Remove external event source" << std::endl;
      Scheduler::remove_external_event_source();
      Cown::release(ThreadAlloc::get(), e);
    });
  });

  sched.run();
  thr.join();
}
