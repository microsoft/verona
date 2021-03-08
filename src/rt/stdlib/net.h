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
      auto* msg = poller.create_msg(ThreadAlloc::get(), event);
      Scheduler::local()->add_blocking_io(msg);
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

    static Result<TCPSocket*>
    connect(Alloc* alloc, const char* host, uint16_t port)
    {
      auto res = TCP::connect(host, port);
      if (!res.ok())
        return res.forward_err<TCPSocket*>();

      return create(alloc, Scheduler::local()->get_io_poller(), *res);
    }

    static Result<TCPSocket*>
    listen(Alloc* alloc, const char* host, uint16_t port)
    {
      auto res = TCP::listen(host, port);
      if (!res.ok())
        return res.forward_err<TCPSocket*>();

      return create(alloc, Scheduler::local()->get_io_poller(), *res);
    }

    Result<TCPSocket*> accept(Alloc* alloc)
    {
      auto res = TCP::accept(event, nullptr);
      if (!res.ok())
      {
        would_block();
        return res.forward_err<TCPSocket*>();
      }
      return create(alloc, poller, *res);
    }

    Result<size_t> read(char* buf, size_t len)
    {
      auto res = TCP::read(event, buf, len);
      if (!res.ok())
        would_block();

      return res;
    }

    Result<size_t> write(char* buf, size_t len)
    {
      auto res = TCP::write(event, buf, len);
      if (!res.ok())
        would_block();
      else
        assert(*res == len);

      return res;
    }

    Result<bool> close()
    {
      assert(!closed);

      closed = true;
      Systematic::cout() << "Close on IO cown " << this << std::endl;
      auto res = TCP::close(event);
      if (!res.ok())
      {
        Systematic::cout() << "Socket close error: " << res.error()
                           << std::endl;
      }

      const auto prev = poller.remove_event_source();
      if (prev == 1)
        Scheduler::remove_external_event_source();

      return res;
    }
  };
}
