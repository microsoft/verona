// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "process_sandbox/cxxsandbox.h"
#include "process_sandbox/sandbox.h"

#include <limits.h>
#include <limits>
#include <stdio.h>
#include <thread>
#include <unistd.h>

using namespace sandbox;
/**
 * Sandboxed function that fetches the contents of a URL.
 */
std::pair<char*, size_t> fetch(char* url);

/**
 * The structure that represents an instance of the sandbox.
 */
struct CurlSandbox
{
  /**
   * The library that defines the functions exposed by this sandbox.
   */
  Library lib = {SANDBOX_LIBRARY};
#define EXPORTED_FUNCTION(public_name, private_name) \
  decltype(make_sandboxed_function<decltype(private_name)>(lib)) public_name = \
    make_sandboxed_function<decltype(private_name)>(lib);
  EXPORTED_FUNCTION(fetch, ::fetch)
};

/**
 * Vector of variable-sized sockaddr values.  These are captured when the
 * sandbox calls `getaddrinfo` and then used when the sandbox calls `connect`
 * to ensure that the sandbox can connect only to addresses that are valid for
 * the desired target.
 */
std::vector<std::vector<char>> valid_addrs;

/**
 * Allow the sandbox to perform lookups on http://example.com but no other
 * service / domain combination.
 */
int auth_getaddrinfo(
  const char* hostname,
  const char* servname,
  const addrinfo* hints,
  addrinfo** res)
{
  fprintf(stderr, "host: %s, service: %s\n", hostname, servname);
  if (
    (strcmp(hostname, "example.com") != 0) ||
    ((strcmp(servname, "http") != 0) && (strcmp(servname, "80") != 0)))
  {
    return EAI_FAIL;
  }
  auto ret = getaddrinfo(hostname, servname, hints, res);
  if (ret == 0)
  {
    for (addrinfo* ai = *res; ai != nullptr; ai = ai->ai_next)
    {
      char* bytes = reinterpret_cast<char*>(ai->ai_addr);
      valid_addrs.emplace_back(bytes, bytes + ai->ai_addrlen);
    }
  }
  return ret;
}

/**
 * Allow the sandbox to connect to any of the addresses that the `getaddrinfo`
 * call returned, but no others.
 */
int auth_connect(int s, const sockaddr* addr, socklen_t addrlen)
{
  for (auto& sa : valid_addrs)
  {
    if ((sa.size() == addrlen) && (memcmp(sa.data(), addr, addrlen) == 0))
    {
      fprintf(stderr, "Allowing connection to previously authorised address\n");
      connect(s, addr, addrlen);
      return 0;
    }
  }
  errno = EADDRNOTAVAIL;
  return -1;
}

int main()
{
  // Create the sandbox
  CurlSandbox curl;
  // Allow it restricted network access
  curl.lib.network_policy()
    .register_handler<NetworkPolicy::NetOperation::GetAddrInfo>(
      auth_getaddrinfo);
  curl.lib.network_policy()
    .register_handler<NetworkPolicy::NetOperation::Connect>(auth_connect);
  // Copy the URL into the sandbox.
  auto url = curl.lib.strdup("http://example.com");
  // Fetch the contents of the URL
  auto res = curl.fetch(url);
  // Free the copy of the URL in the sandbox
  curl.lib.free(url);
  fprintf(stderr, "Fetched %zd bytes:\n", res.second);
  // Check that we received something plausible
  SANDBOX_INVARIANT(
    curl.lib.contains(res.first, res.second), "Return value not in sandbox");
  SANDBOX_INVARIANT(
    strnstr(res.first, "Example Domain", res.second) != nullptr,
    "example.com did not return expected result");
  // Write the received text to the standard error for debugging
  write(STDERR_FILENO, res.first, res.second);
  return 0;
}
