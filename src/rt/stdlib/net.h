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
    int fd;
    bool closed = false;

    TCPSocket(Poller& poller_, int fd_) : poller(poller_), fd(fd_)
    {
      const auto prev = poller.add_event_source();
      if (prev == 0)
        Scheduler::add_external_event_source();
    }

    void would_block()
    {
      auto event = TCP::event(fd, this);
      Scheduler::local()->add_blocking_io(event);
      would_block_on_io();
    }

    static TCPSocket* create(Alloc* alloc, Poller& poller, int socket)
    {
      assert(socket != -1);
      auto* cown = new (alloc) TCPSocket(poller, socket);
      Systematic::cout() << "New TCPSocket cown " << cown << std::endl;
      auto event = TCP::event(socket, cown);
      poller.register_event(event);
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
      int socket = TCP::socket_connect(host, port);
      if (socket == -1)
        return nullptr;

      return create(alloc, Scheduler::local()->get_io_poller(), socket);
    }

    static TCPSocket* listen(Alloc* alloc, const char* host, uint16_t port)
    {
      int socket = TCP::socket_listen(host, port);
      if (socket == -1)
        return nullptr;

      return create(alloc, Scheduler::local()->get_io_poller(), socket);
    }

    TCPSocket* accept(Alloc* alloc)
    {
      auto socket = TCP::accept(fd);
      if (socket == -1)
      {
        would_block();
        return nullptr;
      }

      return create(alloc, poller, socket);
    }

    int read(char* buf, uint32_t len)
    {
      int res = TCP::read(fd, buf, len);
      if (res == -1)
        would_block();

      return res;
    }

    int write(char* buf, uint32_t len)
    {
      int res = TCP::write(fd, buf, len);
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
      auto ret = TCP::close(fd);
      if (ret == -1)
      {
        Systematic::cout() << "Socket close error: " << strerrorname_np(errno)
                           << std::endl;
      }

      const auto prev = poller.remove_event_source();
      if (prev == 1)
        Scheduler::remove_external_event_source();

      return ret;
    }
  };
}
