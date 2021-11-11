// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "net-test-helpers.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <limits>
#include <stdio.h>
#include <thread>

using namespace sandbox;
/**
 * The structure that represents an instance of the sandbox.
 */
struct NetSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(public_name, private_name) \
  decltype(make_sandboxed_function<decltype(private_name)>(lib)) public_name = \
    make_sandboxed_function<decltype(private_name)>(lib);
  EXPORTED_FUNCTION(test_network, ::test_network)
  EXPORTED_FUNCTION(test_bind, ::test_bind)
  EXPORTED_FUNCTION(test_connect, ::test_connect)
  EXPORTED_FUNCTION(test_listen, ::test_listen)
  EXPORTED_FUNCTION(test_receive, ::test_receive)
};

/**
 * The number of times `bind` was called by a sandbox.
 */
int bind_calls;

/**
 * The number of times that `connect` was called by a sandbox.
 */
int connect_calls;

/**
 * The port number to use.  We will start here and try higher ones until we
 * find an unused one.
 */
short port = 1024;

/**
 * Check that the address that the sandbox is trying to use is the one that we
 * expect (127.0.0.1, with the port from `port`).
 */
bool check_addr(const sockaddr* addr, socklen_t addrlen)
{
  auto addr_in = reinterpret_cast<const sockaddr_in*>(addr);
  if (
    (addrlen != sizeof(sockaddr_in)) || (ntohs(addr_in->sin_port) != port) ||
    (htonl(addr_in->sin_addr.s_addr) != 0x7f000001))
  {
    errno = EINVAL;
    return false;
  }
  return true;
};

/**
 * Allow binding to a single address.
 */
int auth_bind(int s, const sockaddr* addr, socklen_t addrlen)
{
  bind_calls++;
  if (!check_addr(addr, addrlen))
  {
    return -1;
  }
  int ret = -1;
  // If you run this test rapidly, the OS may not have made the port available
  // again yet, so try a different one.  The sandbox doesn't have to know...
  for (; port < std::numeric_limits<short>::max(); port++)
  {
    sockaddr_in addr_in = loopback_for_port(port);
    if ((ret = bind(s, reinterpret_cast<sockaddr*>(&addr_in), addrlen)) == 0)
    {
      break;
    }
  }
  return ret;
}

/**
 * Allow connecting to a single address.
 */
int auth_connect(int s, const sockaddr* addr, socklen_t addrlen)
{
  connect_calls++;
  if (!check_addr(addr, addrlen))
  {
    return -1;
  }
  connect(s, addr, addrlen);
  return 0;
}

int main()
{
  // Create a sandbox that can do networking.
  NetSandbox serverBox;
  // Allow it to do getaddrinfo to any adderss
  serverBox.lib.network_policy()
    .allow<NetworkPolicy::NetOperation::GetAddrInfo>();
  // Check that it can look up microsoft.com.
  fprintf(stderr, "Testing getaddrinfo\n");
  serverBox.test_network();

  // Check that it can't do bind (default deny):
  fprintf(stderr, "Testing bind\n");
  serverBox.test_bind(true, port);

  // Allow it to do bind and then check that it does:
  serverBox.lib.network_policy()
    .register_handler<NetworkPolicy::NetOperation::Bind>(auth_bind);
  serverBox.test_bind(false, port);

  // Make the sandbox listen on the socket.
  serverBox.test_listen();

  // In a background thread, ask the sandbox to accept two connections, each of
  // which should send it the contents of `msg`.  Return a buffer with each in
  // it.
  std::thread t{[&]() {
    for (int i = 0; i < 2; i++)
    {
      char* recvbuf = serverBox.test_receive(sizeof(msg));
      // Make sure that the received buffer is in the sandbox!
      SANDBOX_INVARIANT(
        serverBox.lib.contains(recvbuf, sizeof(msg)),
        "Return from receive is not in sandbox {:p}",
        recvbuf);
      SANDBOX_INVARIANT(
        memcmp(recvbuf, msg, sizeof(msg)) == 0,
        "Return from receive is not the correct value.");
      serverBox.lib.free(recvbuf);
    }
  }};

  // Try sending a message to the socket from the parent:
  int s = socket(AF_INET, SOCK_STREAM, 0);
  SANDBOX_INVARIANT(s >= 0, "Socket failed");
  sockaddr_in addr = loopback_for_port(port);
  int ret = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  SANDBOX_INVARIANT(ret == 0, "Connect failed");
  send(s, msg, sizeof(msg), 0);

  // Create another sandbox to act as a network client.
  NetSandbox clientBox;
  // Make sure that it can't connect until we authorise it to
  clientBox.test_connect(true, port);
  // Allow it to connect to a specific address.
  clientBox.lib.network_policy()
    .register_handler<NetworkPolicy::NetOperation::Connect>(auth_connect);
  // Check that it can connect and send a message to the server sandbox.
  clientBox.test_connect(false, port);

  SANDBOX_INVARIANT(
    bind_calls == 1,
    "bind should have been called once was called {} times",
    bind_calls);
  SANDBOX_INVARIANT(
    connect_calls == 1,
    "connect should have been called once was called {} times",
    connect_calls);
  t.join();
  fprintf(stderr, "Test done\n");
  return 0;
}
