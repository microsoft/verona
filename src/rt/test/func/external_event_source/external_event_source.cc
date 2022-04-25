// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <verona.h>

using namespace verona::rt;

int buffer[100];

struct ExternalSource;
void enable_notifications(ExternalSource& es);
void disable_notifications(ExternalSource& es);

struct Poller : VCown<Poller>
{
  int fd;
  bool should_schedule_if_notified;
  int empty_count;
  Noticeboard<int> buffer_idx;
  int read;
  std::shared_ptr<ExternalSource> es;

  Poller()
  : should_schedule_if_notified(false), empty_count(0), buffer_idx(0), read(0)
  {}

  void main_poller()
  {
    int val, read_old;
    auto& alloc = ThreadAlloc::get();

    read_old = read;
    while (read <= buffer_idx.peek(alloc))
    {
      val = buffer[read++];

      Logging::cout() << val << Logging::endl;

      if (val == 19)
        return;
    }
    if (read > read_old)
      empty_count = 0;
    else
      empty_count++;

    if (empty_count < 10)
      schedule_lambda(this, [=]() { main_poller(); });
    else
    {
      // add external source and enable notifications
      enable_notifications(*es);
      Scheduler::add_external_event_source();
      should_schedule_if_notified = true;

      // Check if there are new buffers between last checking and enabling
      // notifications
      if (read <= buffer_idx.peek(alloc))
      {
        disable_notifications(*es);
        Scheduler::remove_external_event_source();
        should_schedule_if_notified = false;
        schedule_lambda(this, [=]() { main_poller(); });
      }
    }
  }

  void notified(Object* o)
  {
    Poller* p = reinterpret_cast<Poller*>(o);

    if (should_schedule_if_notified)
    {
      // No need to disable notifications here because the external source
      // delivers a single-shot notification
      should_schedule_if_notified = false;
      Scheduler::remove_external_event_source();
      schedule_lambda(p, [=]() { p->main_poller(); });
    }
  }
};

struct ExternalSource
{
  Poller* p;
  std::atomic<bool> notifications_on;

  ExternalSource(Poller* p_) : p(p_), notifications_on(false)
  {
    Cown::acquire(p);
  }

  void main_es()
  {
    auto& alloc = ThreadAlloc::get();

    for (int i = 0; i < 10; i++)
    {
      buffer[i] = i;
      p->buffer_idx.update(alloc, i);
    }

    if (notifications_on.exchange(false))
      p->mark_notify();

    // sleep
    auto pause_time = std::chrono::milliseconds(1000);
    std::this_thread::sleep_for(pause_time);

    for (int i = 10; i < 20; i++)
    {
      buffer[i] = i;
      p->buffer_idx.update(alloc, i);
    }

    if (notifications_on.exchange(false))
      p->mark_notify();

    Cown::release(alloc, p);
  }

  void notifications_enable()
  {
    notifications_on = true;
  }

  void notifications_disable()
  {
    notifications_on = false;
  }
};

void enable_notifications(ExternalSource& es)
{
  es.notifications_enable();
}

void disable_notifications(ExternalSource& es)
{
  es.notifications_disable();
}

void test(SystematicTestHarness* harness)
{
  auto& alloc = ThreadAlloc::get();
  auto* p = new (alloc) Poller();
  auto es = std::make_shared<ExternalSource>(p);

  p->es = es;
  schedule_lambda<YesTransfer>(p, [=]() { p->main_poller(); });

  harness->external_thread([=]() { es->main_es(); });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test, &harness);
  return 0;
}
