#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;

struct Ping : public VBehaviour<Ping>
{
  io::TCPSocket* conn;
  bool start = true;

  Ping(io::TCPSocket* conn_) : conn(conn_) {}

  void f()
  {
    auto* alloc = ThreadAlloc::get();
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", "ping");
    if (start)
    {
      int ret = conn->socket_write(alloc, buf, strlen(buf) + 1);
      if (ret == -1)
      {
        perror("ping start");
        std::cout << "Connection closed: " << conn << std::endl;
        Cown::release(alloc, conn);
        return;
      }
      Cown::schedule<Ping>(conn, conn);
      return;
    }

    // char buf[64];
    // int ret = conn->socket_read(alloc, buf, 64);
    // if (ret > 0)
    // {
    //   conn->socket_write(alloc, ping, strlen(ping));
    // }
    // else if (ret == 0)
    // {
    //   std::cout << "Connection closed: " << conn << std::endl;
    //   Cown::release(alloc, conn);
    //   return;
    // }

    // Cown::schedule<Ping>(conn, conn);
  }
};

struct Pong : public VBehaviour<Pong>
{
  io::TCPSocket* conn;

  Pong(io::TCPSocket* conn_) : conn(conn_) {}

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
  io::TCPSocket* listener;

  Listen(io::TCPSocket* listener_) : listener(listener_) {}

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

struct Main : public VCown<Main>
{};

struct Init : public VBehaviour<Init>
{
  uint16_t server_port;

  Init(uint16_t server_port_) : server_port(server_port_) {}

  void f()
  {
    auto* alloc = ThreadAlloc::get_noncachable();

    // auto* listener = io::TCPSocket::listen(alloc, "", server_port);
    // Cown::schedule<Listen>(listener, listener);

    auto* client = io::TCPSocket::connect(alloc, "", server_port);
    Cown::schedule<Ping>(client, client);
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
