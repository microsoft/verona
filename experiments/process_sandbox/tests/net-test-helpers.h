// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
bool test_network();
bool test_bind(bool, short);
bool test_connect(bool, short);
bool test_send();
void test_listen();
char* test_receive(size_t);

/**
 * Helper to set up a loopback IPv4 address for a specific port.
 */
static sockaddr_in loopback_for_port(short port)
{
  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(0x7f000001);
  addr.sin_port = htons(port);
  return addr;
}

/**
 * Test message to send over a socket.
 */
const char msg[] = "Hello World!";
