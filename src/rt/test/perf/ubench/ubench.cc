// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * A microbenchmark for measuring message passing rates in the Verona runtime.
 * This microbenchmark is adapted from the original message-ubench from the Pony
 * Language examples to include multi-messages.
 *
 * This microbenchmark executes a sequence of intervals that are 1 second long
 * by default. During each interval the `Monitor` cown and a static set of
 * `Pinger` cowns are setup and an initial set of `Ping` messages are sent to
 * the `Pinger`s. When a `Pinger` receives a `Ping` message, the `Pinger` will
 * randomly choose another `Pinger` to forward the `Ping` message. A `Pinger`
 * may randomly choose to include itself in the forwarded `Ping` multi-message
 * along with the selected recipient. By default 5% of `Ping` messages will
 * become these multi-messages.
 */

#include "test/log.h"
#include "test/opt.h"
#include "test/xoroshiro.h"
#include "verona.h"

#include <chrono>

namespace sn = snmalloc;
namespace rt = verona::rt;

static rt::Cown** all_cowns = nullptr;
static size_t all_cowns_count = 0;

struct Pinger : public rt::VCown<Pinger>
{
  vector<Pinger*>& pingers;
  xoroshiro::p128r32 rng;
  size_t select_mod = 0;
  bool running = false;
  size_t count = 0;

  Pinger(vector<Pinger*>& pingers_, size_t seed, size_t percent_multimessage)
  : pingers(pingers_), rng(seed)
  {
    if (percent_multimessage != 0)
      select_mod = (size_t)((double)100.00 / (double)percent_multimessage);
  }
};

struct Monitor : public rt::VCown<Monitor>
{
  vector<Pinger*>& pingers;
  size_t initial_pings;
  std::chrono::seconds report_interval;
  size_t report_count;
  size_t waiting = 0;
  uint64_t start = 0;

  Monitor(
    vector<Pinger*>& pingers_,
    size_t initial_pings_,
    std::chrono::seconds report_interval_,
    size_t report_count_)
  : pingers(pingers_),
    initial_pings(initial_pings_),
    report_interval(report_interval_),
    report_count(report_count_)
  {}

  void trace(rt::ObjectStack& st) const
  {
    for (auto* p : pingers)
      st.push(p);
  }
};

struct Ping : public rt::VBehaviour<Ping>
{
  Pinger* pinger;
  std::array<Pinger*, 2> recipients;

  Ping(Pinger* pinger_) : pinger(pinger_) {}

  void f()
  {
    if (!pinger->running)
      return;

    pinger->count++;

    size_t cowns = 1;
    recipients[1] = pinger;
    recipients[0] =
      pinger->pingers[pinger->rng.next() % pinger->pingers.size()];

    if ((pinger->pingers.size() > 1) && (pinger->select_mod != 0))
    {
      while (recipients[0] == pinger)
      {
        recipients[0] =
          pinger->pingers[pinger->rng.next() % pinger->pingers.size()];
      }
      if ((pinger->rng.next() % pinger->select_mod) == 0)
        cowns = 2;
    }

    rt::Cown::schedule<Ping>(
      cowns, (rt::Cown**)recipients.data(), recipients[0]);
  }
};

struct Stop;
struct StopPinger;
struct NotifyStopped;
struct Report;

static void start_timer(Monitor* monitor, std::chrono::milliseconds timeout)
{
  rt::Cown::acquire(monitor);
  rt::Scheduler::set_allow_teardown(false);
  std::thread([=]() mutable {
    std::this_thread::sleep_for(timeout);
    rt::Cown::schedule<Stop, rt::YesTransfer>((rt::Cown*)monitor, monitor);
    rt::Scheduler::set_allow_teardown(true);
  }).detach();
}

struct Start : public rt::VBehaviour<Start>
{
  Monitor* monitor;

  Start(Monitor* monitor_) : monitor(monitor_) {}

  void f()
  {
    for (auto* p : monitor->pingers)
    {
      p->count = 0;
      p->running = true;
      for (size_t i = 0; i < monitor->initial_pings; i++)
        rt::Cown::schedule<Ping>(p, p);
    }

    monitor->start = sn::Aal::tick();
    start_timer(monitor, monitor->report_interval);
  }
};

struct Stop : public rt::VBehaviour<Stop>
{
  Monitor* monitor;

  Stop(Monitor* monitor_) : monitor(monitor_) {}

  void f()
  {
    monitor->waiting = monitor->pingers.size();
    for (auto* pinger : monitor->pingers)
      rt::Cown::schedule<StopPinger>(pinger, pinger, monitor);
  }
};

struct StopPinger : public rt::VBehaviour<StopPinger>
{
  Pinger* pinger;
  Monitor* monitor;

  StopPinger(Pinger* pinger_, Monitor* monitor_)
  : pinger(pinger_), monitor(monitor_)
  {}

  void f()
  {
    pinger->running = false;
    rt::Cown::schedule<NotifyStopped>(monitor, monitor);
  }
};

struct NotifyStopped : public rt::VBehaviour<NotifyStopped>
{
  Monitor* monitor;

  NotifyStopped(Monitor* monitor_) : monitor(monitor_) {}

  void f()
  {
    if (--monitor->waiting != 0)
      return;

    rt::Cown::schedule<Report>(all_cowns_count, all_cowns, monitor);

    if (--monitor->report_count != 0)
      rt::Cown::schedule<Start>(all_cowns_count, all_cowns, monitor);
    else
      rt::Cown::release(sn::ThreadAlloc::get(), monitor);
  }
};

struct Report : public rt::VBehaviour<Report>
{
  Monitor* monitor;

  Report(Monitor* monitor_) : monitor(monitor_) {}

  void trace(rt::ObjectStack& st) const
  {
    st.push(monitor);
  }

  void f()
  {
    uint64_t t = sn::Aal::tick() - monitor->start;
    uint64_t sum = 0;
    for (auto* p : monitor->pingers)
      sum += p->count;

    uint64_t rate = (sum * 1'000'000'000) / t;
    logger::cout() << t << " ns, " << rate << " msgs/s" << std::endl;
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  const auto seed = opt.is<size_t>("--seed", 5489);
  const auto cores = opt.is<size_t>("--cores", 4);
  const auto pingers = opt.is<size_t>("--pingers", 8);
  const auto report_interval =
    std::chrono::seconds(opt.is<size_t>("--report_interval", 1));
  const auto report_count = opt.is<size_t>("--report_count", 10);
  const auto initial_pings = opt.is<size_t>("--initial_pings", 5);
  const auto percent_multimessage = opt.is<size_t>("--percent_multimessage", 5);
  assert(percent_multimessage <= 100);

  logger::cout() << "cores: " << cores << ", pingers: " << pingers
                 << ", report_interval: " << report_interval.count()
                 << ", initial_pings: " << initial_pings
                 << ", percent_mutlimessage: " << percent_multimessage
                 << std::endl;

  auto* alloc = sn::ThreadAlloc::get();
#ifdef USE_SYSTEMATIC_TESTING
  Systematic::enable_logging();
  Systematic::set_seed(seed);
#else
  UNUSED(seed);
#endif
  auto& sched = rt::Scheduler::get();
  sched.set_fair(true);
  sched.init(cores);

  static vector<Pinger*> pinger_set;
  for (size_t p = 0; p < pingers; p++)
    pinger_set.push_back(new (alloc)
                           Pinger(pinger_set, seed + p, percent_multimessage));

  auto* monitor = new (alloc)
    Monitor(pinger_set, initial_pings, report_interval, report_count);

  all_cowns_count = pingers + 1;
  all_cowns = (rt::Cown**)alloc->alloc(all_cowns_count * sizeof(rt::Cown*));
  memcpy(all_cowns, pinger_set.data(), pinger_set.size() * sizeof(rt::Cown*));
  all_cowns[pinger_set.size()] = monitor;

  rt::Cown::schedule<Start>(all_cowns_count, all_cowns, monitor);

  sched.run();
  alloc->dealloc(all_cowns, all_cowns_count * sizeof(rt::Cown*));
  return 0;
}
