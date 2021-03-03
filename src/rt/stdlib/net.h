#pragma once

#include "../cpp/vobject.h"
#include "../pal/pal.h"
#include "../sched/cown.h"
#include "../sched/schedulerthread.h"

namespace verona::rt::io
{
  class TCPSock : public rt::VCown<TCPSock>
  {
  private:
    int fd;

    TCPSock(int fd_) : fd(fd_) {}

    void would_block()
    {
      auto& io_poller = Scheduler::local()->get_io_poller();
      io_poller.socket_notify(fd, this);
      Scheduler::local()->add_io_source();
      would_block_on_io();
    }

  public:
    int socket_read(char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_read(fd, buf, len);
      if (res == -1)
        would_block();

      return res;
    }

    int socket_write(char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_write(fd, buf, len);
      if (res == -1)
        would_block();

      assert(static_cast<uint32_t>(res) == len);
      return res;
    }

    TCPSock* server_accept()
    {
      int sock = DefaultTCPSocket::server_accept(fd);
      if (sock == -1)
      {
        Systematic::cout() << "TCP accept error: " << strerror(errno)
                           << std::endl;
        would_block();
        return nullptr;
      }
      DefaultTCPSocket::socket_config(sock);

      auto* alloc = rt::ThreadAlloc::get();
      auto* cown = new (alloc) TCPSock(sock);
      Systematic::cout() << "New TCP connection cown " << cown << std::endl;
      return cown;
    }

    static TCPSock* server_listen(Alloc* alloc, uint16_t port)
    {
      int sock = DefaultTCPSocket::server_listen(port);
      if (sock < 0)
        return nullptr;

      auto* cown = new (alloc) TCPSock(sock);
      Systematic::cout() << "New TCP listener cown " << cown << std::endl;
      return cown;
    }
  };
}
