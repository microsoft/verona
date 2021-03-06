#include "client.h"
#include "server.h"

#include <test/opt.h>
#include <verona.h>

using namespace verona::rt;

struct Main : public VCown<Main>
{};

struct Init : public VBehaviour<Init>
{
  uint16_t server_port;
  uint16_t client_port;

  Init(uint16_t server_port_, uint16_t client_port_)
  : server_port(server_port_), client_port(client_port_)
  {}

  void f()
  {
    auto* alloc = ThreadAlloc::get_noncachable();

    auto* listener = io::TCPSock::server_listen(alloc, server_port);
    Cown::schedule<Listen>(listener, listener);

    // auto* client = io::TCPSock::
  }
};

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  const auto seed = opt.is<size_t>("--seed", 5489);
  const auto cores = opt.is<size_t>("--cores", 4);
  const auto server = opt.is<uint16_t>("--server", 8080);
  const auto client = opt.is<uint16_t>("--client", 8081);

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
  Cown::schedule<Init>(entrypoint, server, client);
  Cown::release(alloc, entrypoint);

  sched.run();
}
