// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "net-test-helpers.h"
#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <stdio.h>

/**
 * Test that getaddrinfo works inside a network-enabled sandbox.
 */
bool test_network()
{
  addrinfo* res;
  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  fprintf(stderr, "Looking up host\n");
  int ret = getaddrinfo("microsoft.com", "http", &hints, &res);
  SANDBOX_INVARIANT(ret == 0, "getaddrinfo returned {}", gai_strerror(ret));
  auto port = ((sockaddr_in*)res->ai_addr)->sin_port;
  SANDBOX_INVARIANT(
    port == htons(80), "HTTP should be port 80, was {}", ntohs(port));
  int s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  SANDBOX_INVARIANT(s >= 0, "Failed to create socket {}", strerror(errno));
  ret = connect(s, res->ai_addr, res->ai_addrlen);
  SANDBOX_INVARIANT(
    ret == -1,
    "connect() returned {}, the policy should have blocked it and generated a "
    "-1 return value",
    ret);
  return ret == 0;
}

/**
 * Socket.  This sandbox uses global (sandbox-local) state to store the socket
 * that it uses for testing.
 */
int s;

/**
 * Try to bind a socket to a specific port.  If `expectFailure` is true then
 * this tests that it *can't* bind the socket (i.e. when the sandbox policy
 * does not allow it).
 */
bool test_bind(bool expectFailure, short port)
{
  s = socket(AF_INET, SOCK_STREAM, 0);
  SANDBOX_INVARIANT(s >= 0, "Failed to create socket {}", strerror(errno));
  sockaddr_in addr = loopback_for_port(port);
  int ret = bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  if (expectFailure)
  {
    SANDBOX_INVARIANT(
      ret == -1, "bind succeeded when it should have been blocked");
  }
  else
  {
    SANDBOX_INVARIANT(ret == 0, "bind failed {}", strerror(errno));
  }
  return true;
}

/**
 * Try to connect a socket to a specific remote address and port and send it a
 * message.  If `expectFailure` is true then this tests that it *can't* connect
 * the socket (i.e. when the sandbox policy does not allow it).
 */
bool test_connect(bool expectFailure, short port)
{
  s = socket(AF_INET, SOCK_STREAM, 0);
  SANDBOX_INVARIANT(s >= 0, "Failed to create socket {}", strerror(errno));
  sockaddr_in addr = loopback_for_port(port);
  int ret = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  if (expectFailure)
  {
    SANDBOX_INVARIANT(
      ret == -1, "connect succeeded when it should have been blocked");
  }
  else
  {
    SANDBOX_INVARIANT(ret == 0, "connect failed {}", strerror(errno));
    send(s, msg, sizeof(msg), 0);
    close(s);
  }
  return true;
}

/**
 * Test that we can listen.  Listening does not affect a global namespace and
 * so the sandbox policy *should* allow this unconditionally.
 */
void test_listen()
{
  int ret = listen(s, 2);
  SANDBOX_INVARIANT(ret == 0, "listen returned failure: {}", strerror(errno));
}

/**
 * Test accepting a connection and receiving a message.  These operations do
 * not manipulate a global namespace and so should all work unconditionally, as
 * long as the previous steps were authorised.
 */
char* test_receive(size_t size)
{
  char* buf = new char[size]();
  fprintf(stderr, "Sandbox blocking on accept\n");
  int conn = accept(s, nullptr, nullptr);
  fprintf(stderr, "Sandbox accepted connection\n");
  SANDBOX_INVARIANT(conn >= 0, "accept failed: {}", strerror(errno));
  auto ret = recv(conn, buf, size, MSG_WAITALL);
  SANDBOX_INVARIANT(
    ret == static_cast<ssize_t>(size),
    "recv returned {} (expected {}), errno: {}",
    ret,
    size,
    strerror(errno));
  return buf;
}

extern "C" void sandbox_init()
{
  sandbox::ExportedLibrary::export_function(::test_network);
  sandbox::ExportedLibrary::export_function(::test_bind);
  sandbox::ExportedLibrary::export_function(::test_connect);
  sandbox::ExportedLibrary::export_function(::test_listen);
  sandbox::ExportedLibrary::export_function(::test_receive);
}
