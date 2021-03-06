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

    TCPSocket(io::DefaultPoller<Cown>& poller_, int fd_)
    : poller(poller_), fd(fd_)
    {
      Scheduler::local()->add_io_source();
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
      const auto local = &Scheduler::local()->get_io_poller() == &poller;
      poller.socket_deregister(ThreadAlloc::get(), fd, local);
      Scheduler::local()->remove_io_source();
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

    TCPSocket* server_accept(Alloc* alloc)
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

    int socket_read(Alloc* alloc, char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_read(fd, buf, len);
      if (res == -1)
        would_block(alloc);

      return res;
    }

    int socket_write(Alloc* alloc, char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_write(fd, buf, len);
      if (res == -1)
        would_block(alloc);
      else
        assert((uint32_t)res == len);

      return res;
    }
  };
}
