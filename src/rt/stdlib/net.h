#pragma once

#include "../cpp/vobject.h"
#include "../pal/pal.h"
#include "../sched/cown.h"
#include "../sched/schedulerthread.h"

namespace verona::rt::io
{
  class TCPSocket : public rt::VCown<TCPSocket>
  {
  private:
    Poller& poller;
    Event event;
    bool closed = false;

    TCPSocket(Poller& poller_, Event event_) : poller(poller_), event(event_)
    {
      event.set_cown(this);

      const auto prev = poller.add_event_source();
      if (prev == 0)
        Scheduler::add_external_event_source();
    }

    void would_block()
    {
      Scheduler::local()->add_blocking_io(event);
      would_block_on_io();
    }

    static inline TCPSocket* create(Alloc* alloc, Poller& poller, Event event)
    {
      auto* cown = new (alloc) TCPSocket(poller, std::move(event));
      Systematic::cout() << "New TCPSocket cown " << cown << std::endl;
      poller.register_event(cown->event);
      return cown;
    }

  public:
    ~TCPSocket()
    {
      if (!closed)
        close();
    }

    static TCPSocket* connect(Alloc* alloc, const char* host, uint16_t port)
    {
      auto res = TCP::connect(host, port);
      if (!res)
        return nullptr;

      return create(alloc, Scheduler::local()->get_io_poller(), *res);
    }

    static TCPSocket* listen(Alloc* alloc, const char* host, uint16_t port)
    {
      auto res = TCP::listen(host, port);
      if (!res)
        return nullptr;

      return create(alloc, Scheduler::local()->get_io_poller(), *res);
    }

    TCPSocket* accept(Alloc* alloc)
    {
      auto res = TCP::accept(event, nullptr);
      if (!res)
      {
        would_block();
        return nullptr;
      }
      return create(alloc, poller, *res);
    }

    int read(char* buf, uint32_t len)
    {
      int res = TCP::read(event, buf, len);
      if (res == -1)
        would_block();

      return res;
    }

    int write(char* buf, uint32_t len)
    {
      int res = TCP::write(event, buf, len);
      if (res == -1)
        would_block();
      else
        assert(static_cast<uint32_t>(res) == len);

      return res;
    }

    int close()
    {
      assert(!closed);

      closed = true;
      Systematic::cout() << "Close on IO cown " << this << std::endl;
      auto res = TCP::close(event);
      if (res == -1)
      {
        Systematic::cout() << "Socket close error: " << strerrorname_np(errno)
                           << std::endl;
      }

      const auto prev = poller.remove_event_source();
      if (prev == 1)
        Scheduler::remove_external_event_source();

      return res;
    }
  };
}
