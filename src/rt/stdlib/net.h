#pragma once

#include "../cpp/vobject.h"
#include "../pal/pal.h"

namespace verona::rt::io
{
  class TCPSock : public rt::VCown<TCPSock>
  {
  private:
    int fd;

    TCPSock(int fd_) : fd(fd_)
    {
      Scheduler::local()->add_io_source();
    }

  public:
    void dispose()
    {
      DefaultPoller::unregister_socket(Scheduler::local()->io_fd(), fd);
      Scheduler::local()->remove_io_source();
    }

    int socket_read(char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_read(fd, buf, len);
      if (res == -1)
        would_block_in_io();

      return res;
    }

    int socket_write(char* buf, uint32_t len)
    {
      int res = DefaultTCPSocket::socket_write(fd, buf, len);
      if (res == -1)
        would_block_in_io();

      assert(static_cast<uint32_t>(res) == len);
      return res;
    }

    TCPSock* server_accept()
    {
      int sock = DefaultTCPSocket::server_accept(fd);
      if (sock == -1)
      {
        Systematic::cout() << "TCP accept: " << strerror(errno) << std::endl;
        would_block_in_io();
        return nullptr;
      }
      DefaultTCPSocket::socket_config(sock);

      auto* alloc = rt::ThreadAlloc::get();
      const auto io_fd = Scheduler::local()->io_fd();
      auto* cown = new (alloc) TCPSock(sock);
      DefaultPoller::register_socket(io_fd, sock, 0, cown);
      Systematic::cout() << "New TCP connection cown " << cown << std::endl;
      return cown;
    }

    static TCPSock* server_listen(Alloc* alloc, uint16_t port)
    {
      int sock = DefaultTCPSocket::server_listen(port);
      if (sock < 0)
        return nullptr;

      const auto io_fd = Scheduler::local()->io_fd();
      auto* cown = new (alloc) TCPSock(sock);
      DefaultPoller::register_socket(io_fd, sock, 0, cown);
      Systematic::cout() << "New TCP listener cown " << cown << std::endl;
      return cown;
    }
  };
}
