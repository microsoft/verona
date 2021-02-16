#pragma once

#include "../cpp/vobject.h"
#include "../pal/pal.h"

namespace verona::rt
{
  class TCPSock : public rt::VCown<TCPSock>
  {
  private:
    using Scheduler = rt::ThreadPool<rt::SchedulerThread<rt::Cown>>;

    int fd;
    TCPSock(int fd_) : fd(fd_) {}

  public:
    int socket_read(char* buf, uint32_t len)
    {
      int res;

      res = DefaultTCPSocket::socket_read(fd, buf, len);
      if (res == -1)
        would_block_in_io();

      return res;
    }

    int socket_write(char* buf, uint32_t len)
    {
      int res;

      res = DefaultTCPSocket::socket_write(fd, buf, len);
      if (res == -1)
        would_block_in_io();

      assert(static_cast<uint32_t>(res) == len);
      return res;
    }

    TCPSock* server_accept()
    {
      int new_sock;

      new_sock = DefaultTCPSocket::server_accept(fd);
      if (new_sock == -1)
      {
        would_block_in_io();
        return nullptr;
      }

      DefaultTCPSocket::socket_config(new_sock);

      auto* alloc = rt::ThreadAlloc::get();
      auto sock_cown = new (alloc) TCPSock(new_sock);
      Cown::acquire(sock_cown);
      Scheduler::local()->register_socket(new_sock, 0, (long)(void*)sock_cown);

      return sock_cown;
    }

    // -1 Error, 0 not yet, 1 success
    int check_connected()
    {
      return 0;
    }

    static TCPSock* server_listen(int16_t port)
    {
      int sock;
      std::cout << "Call server listen" << std::endl;

      sock = DefaultTCPSocket::server_listen(port);
      if (sock < 0)
        return nullptr;

      auto* alloc = rt::ThreadAlloc::get();
      auto sock_cown = new (alloc) TCPSock(sock);
      Cown::acquire(sock_cown);
      Scheduler::local()->register_socket(sock, 0, (long)(void*)sock_cown);

      std::cout << "Created new server socket" << std::endl;
      return sock_cown;
    }

    static TCPSock* client_dial(char *ip, uint16_t port)
    {
      int sock;

      std::cout << "Call client dial" << std::endl;
      sock = DefaultTCPSocket::client_dial(ip, port);
      if (sock < 0)
        return nullptr;

      auto* alloc = rt::ThreadAlloc::get();
      auto sock_cown = new (alloc) TCPSock(sock);
      Cown::acquire(sock_cown);
      Scheduler::local()->register_socket(sock, 0, (long)(void*)sock_cown);

      return nullptr;
    }
  };
}
