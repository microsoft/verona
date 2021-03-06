#include <iostream>
#include <verona.h>

using namespace verona::rt;

struct Pong : public VBehaviour<Pong>
{
  io::TCPSock* conn;

  Pong(io::TCPSock* conn_) : conn(conn_) {}

  void f()
  {
    auto* alloc = ThreadAlloc::get();
    char buf[64];
    int ret = conn->socket_read(alloc, buf, 64);
    if (ret > 0)
    {
      conn->socket_write(alloc, buf, (uint32_t)ret);
    }
    else if (ret == 0)
    {
      std::cout << "Connection closed: " << conn << std::endl;
      Cown::release(alloc, conn);
      return;
    }

    Cown::schedule<Pong>(conn, conn);
  }
};

struct Listen : public VBehaviour<Listen>
{
  io::TCPSock* listener;

  Listen(io::TCPSock* listener_) : listener(listener_) {}

  void f()
  {
    auto* conn = listener->server_accept(ThreadAlloc::get_noncachable());
    if (conn != nullptr)
    {
      std::cout << "Received new connection: " << conn << std::endl;
      Cown::schedule<Pong>(conn, conn);
      Cown::release(ThreadAlloc::get(), listener);
      return;
    }

    Cown::schedule<Listen>(listener, listener);
  }
};
