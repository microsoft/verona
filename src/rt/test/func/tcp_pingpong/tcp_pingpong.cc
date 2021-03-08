#include <test/harness.h>
#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;

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
      int ret = conn->write(buf, strlen(buf) + 1);
      assert(ret > 0);
      Cown::schedule<Ping>(conn, conn, false);
      return;
    }

    int ret = conn->read(buf, buf_len);
    if (ret == 0)
    {
      std::cout << "Server connection closed: " << conn << std::endl;
      return;
    }
    else if ((ret == -1) && (errno != EAGAIN) && (errno != EWOULDBLOCK))
    {
      std::cout << "Client read error: " << strerrorname_np(errno) << std::endl;
      return;
    }
    else if (ret > 0)
    {
      std::cout << "Client recv: " << buf << std::endl;
      std::string ping = "ping";
      ret = conn->write((char*)ping.c_str(), ping.length() + 1);
      assert(ret > 0);

      if (Systematic::coin(4))
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
    auto ret = conn->read(buf, buf_len);
    if (ret == 0)
    {
      std::cout << "Client connection closed: " << conn << std::endl;
      return;
    }
    else if ((ret == -1) && (errno != EAGAIN) && (errno != EWOULDBLOCK))
    {
      std::cout << "Server read error: " << strerrorname_np(errno) << std::endl;
      return;
    }
    else if (ret > 0)
    {
      std::cout << "Server recv: " << buf << std::endl;
      std::string pong = "pong";
      ret = conn->write((char*)pong.c_str(), pong.length() + 1);
      assert(ret > 0);
      if (Systematic::coin(4))
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
    if (first_run)
    {
      std::cout << "Server listening" << std::endl;
      first_run = false;

      auto* alloc = ThreadAlloc::get_noncachable();
      auto* client = io::TCPSocket::connect(alloc, "", port);
      if (client == nullptr)
      {
        std::cout << "Unable to connect" << std::endl;
        abort();
      }
      Cown::schedule<Ping, YesTransfer>(client, client);
    }

    auto* conn = listener->accept(ThreadAlloc::get_noncachable());
    if (conn != nullptr)
    {
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
    auto* listener = io::TCPSocket::listen(alloc, "", port);
    if (listener == nullptr)
    {
      std::cout << "Unable to listen" << std::endl;
      return;
    }
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
