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
    io::DefaultPoller<Cown>& poller;
    int fd;
    bool closed = false;

    TCPSocket(io::DefaultPoller<Cown>& poller_, int fd_)
    : poller(poller_), fd(fd_)
    {
      const auto prev = poller.add_event_source();
      if (prev == 0)
        Scheduler::add_external_event_source();
    }

    void would_block(Alloc* alloc)
    {
      const auto local = &Scheduler::local()->get_io_poller() == &poller;
      poller.socket_rearm(alloc, fd, this, local);
      would_block_on_io();
    }

  public:
    ~TCPSocket()
    {
      if (!closed)
        close();
    }

    static TCPSocket* connect(Alloc* alloc, const char* host, uint16_t port)
    {
      int socket = DefaultTCPSocket::socket_connect(host, port);
      if (socket == -1)
        return nullptr;

      auto& poller = Scheduler::local()->get_io_poller();
      auto* cown = new (alloc) TCPSocket(poller, socket);
      Systematic::cout() << "New TCP connection cown " << cown << std::endl;
      poller.socket_register(socket, cown);
      return cown;
    }

    static TCPSocket* listen(Alloc* alloc, const char* host, uint16_t port)
    {
      int socket = DefaultTCPSocket::socket_listen(host, port);
      if (socket == -1)
        return nullptr;

      auto& poller = Scheduler::local()->get_io_poller();
      auto* cown = new (alloc) TCPSocket(poller, socket);
      Systematic::cout() << "New TCP listener cown " << cown << std::endl;
      poller.socket_register(socket, cown);
      return cown;
    }

    TCPSocket* accept(Alloc* alloc)
    {
      int socket = DefaultTCPSocket::server_accept(fd);
      if (socket == -1)
      {
        Systematic::cout() << "TCP accept error: " << strerror(errno)
                           << std::endl;
        would_block(alloc);
        return nullptr;
      }
      DefaultTCPSocket::make_nonblocking(socket);

      auto* cown = new (alloc) TCPSocket(poller, socket);
      Systematic::cout() << "New TCP connection cown " << cown << std::endl;
      poller.socket_register(socket, cown);
      return cown;
    }

    int read(Alloc* alloc, char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_read(fd, buf, len);
      if (res == -1)
        would_block(alloc);

      return res;
    }

    int write(Alloc* alloc, char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_write(fd, buf, len);
      if (res == -1)
        would_block(alloc);
      else
        assert(static_cast<uint32_t>(res) == len);

      return res;
    }

    int close()
    {
      assert(!closed);

      closed = true;
      Systematic::cout() << "Close on IO cown " << this << std::endl;
      auto ret = DefaultTCPSocket::close(fd);
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
