#include <test/harness.h>
#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;

static bool coin(size_t n)
{
#ifdef USE_SYSTEMATIC_TESTING
  return Systematic::coin(n);
#else
  abort();
#endif
}

struct Ping : public VBehaviour<Ping>
{
  io::TCPSocket* conn;
  bool start;

  Ping(io::TCPSocket* conn_, bool start_ = true) : conn(conn_), start(start_) {}

  void f()
  {
    static constexpr size_t buf_len = 64;
    char buf[buf_len];
    snprintf(buf, sizeof(buf), "%s", "ping");
    if (start)
    {
      const auto res = conn->write(buf, strlen(buf) + 1);
      assert(res.ok());
      Cown::schedule<Ping>(conn, conn, false);
      return;
    }

    auto res = conn->read(buf, buf_len);
    if (!res.ok())
    {
      if (!res.would_block())
      {
        std::cout << "Client read error: " << res.error() << std::endl;
        conn->close();
        return;
      }
    }
    else if (*res == 0)
    {
      std::cout << "Server connection closed: " << conn << std::endl;
      return;
    }
    else
    {
      std::cout << "Client recv: " << buf << std::endl;
      std::string ping = "ping";
      res = conn->write((char*)ping.c_str(), ping.length() + 1);
      assert(res.ok());

      if (coin(4))
      {
        conn->close();
        return;
      }
    }
    Cown::schedule<Ping>(conn, conn, false);
  }
};

struct Pong : public VBehaviour<Pong>
{
  io::TCPSocket* conn;

  Pong(io::TCPSocket* conn_) : conn(conn_) {}

  void f()
  {
    static constexpr size_t buf_len = 64;
    char buf[buf_len];
    auto res = conn->read(buf, buf_len);
    if (!res.ok())
    {
      if (!res.would_block())
      {
        std::cout << "Server read error: " << res.error() << std::endl;
        conn->close();
        return;
      }
    }
    else if (*res == 0)
    {
      std::cout << "Client connection closed: " << conn << std::endl;
      return;
    }
    else
    {
      std::cout << "Server recv: " << buf << std::endl;
      std::string pong = "pong";
      res = conn->write((char*)pong.c_str(), pong.length() + 1);
      assert(res.ok());
      if (coin(4))
      {
        conn->close();
        return;
      }
    }
    Cown::schedule<Pong>(conn, conn);
  }
};

struct Listen : public VBehaviour<Listen>
{
  io::TCPSocket* listener;
  uint16_t port;
  bool first_run = true;

  Listen(io::TCPSocket* listener_, uint16_t port_)
  : listener(listener_), port(port_)
  {}

  void f()
  {
    auto* alloc = ThreadAlloc::get();
    if (first_run)
    {
      std::cout << "Server listening" << std::endl;
      first_run = false;

      auto res = io::TCPSocket::connect(alloc, "", port);
      if (!res.ok())
      {
        std::cout << "Unable to connect: " << res.error() << std::endl;
        abort();
      }
      auto* client = *res;
      Cown::schedule<Ping, YesTransfer>(client, client);
    }

    auto res = listener->accept(alloc);
    if (res.ok())
    {
      auto* conn = *res;
      std::cout << "Received new connection: " << conn << std::endl;
      Cown::schedule<Pong, YesTransfer>(conn, conn);
      return;
    }

    Cown::schedule<Listen>(listener, listener, port);
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
    auto res = io::TCPSocket::listen(alloc, "", port);
    if (!res.ok())
    {
      std::cout << "Unable to listen: " << res.error() << std::endl;
      return;
    }
    auto* listener = *res;
    Cown::schedule<Listen, YesTransfer>(listener, listener, port);
  }
};

void test(uint16_t port, bool increment_port)
{
  static uint16_t port_inc = 0;
  if (increment_port)
  {
    port_inc++;
    port += port_inc;
  }

  std::cout << "port: " << port << std::endl;
  auto* alloc = ThreadAlloc::get();
  auto* entrypoint = new (alloc) Main();
  Cown::schedule<Init, YesTransfer>(entrypoint, port);
}

int main(int argc, char** argv)
{
#ifndef USE_SYSTEMATIC_TESTING
  std::cout << "This test requires systematic testing" << std::endl;
  return 1;
#endif

  SystematicTestHarness h(argc, argv);
  const auto port = h.opt.is<uint16_t>("--port", 8080);
  const auto increment_port = h.opt.has("--increment_port");
  h.run(test, port, increment_port);
}
