#include <iostream>
#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;

struct Echo : public VBehaviour<Echo>
{
  io::TCPSock* conn;

  Echo(io::TCPSock* conn_) : conn(conn_) {}

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
      Cown::release(ThreadAlloc::get_noncachable(), conn);
      return;
    }

    Cown::schedule<Echo>(conn, conn);
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
      Cown::schedule<Echo>(conn, conn);
      Cown::release(ThreadAlloc::get(), listener);
      return;
    }

    Cown::schedule<Listen>(listener, listener);
  }
};

struct Main : public VCown<Main>
{};

struct Init : public VBehaviour<Init>
{
  uint16_t port;

  Init(uint16_t port_) : port(port_) {}

  void f()
  {
    auto* alloc = ThreadAlloc::get_noncachable();
    auto* listener = io::TCPSock::server_listen(alloc, port);
    Cown::schedule<Listen>(listener, listener);
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  const auto seed = opt.is<size_t>("--seed", 5489);
  const auto cores = opt.is<size_t>("--cores", 4);
  const auto port = opt.is<uint16_t>("--port", 8080);

#ifdef USE_SYSTEMATIC_TESTING
  Systematic::enable_logging();
  Systematic::set_seed(seed);
#else
  UNUSED(seed);
#endif

  auto& sched = Scheduler::get();
  Scheduler::set_detect_leaks(true);
  sched.set_fair(true);
  sched.init(cores);

  auto* alloc = ThreadAlloc::get();
  auto* entrypoint = new (alloc) Main();
  Cown::schedule<Init>(entrypoint, port);
  Cown::release(alloc, entrypoint);

  sched.run();
}
