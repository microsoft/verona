// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "process_sandbox/platform/platform.h"

using namespace sandbox::platform;

int main(void)
{
  auto sp = SocketPair::create();
  int i = 42;
  SANDBOX_INVARIANT(
    write(sp.first.fd, &i, sizeof(i)) == sizeof(i), "Write failed");
  SANDBOX_INVARIANT(
    read(sp.second.fd, &i, sizeof(i)) == sizeof(i), "Read failed");
  SANDBOX_INVARIANT(i == 42, "Received value {} != 42", i);
  i = 0x12345678;
  SANDBOX_INVARIANT(
    write(sp.second.fd, &i, sizeof(i)) == sizeof(i), "Write failed");
  SANDBOX_INVARIANT(
    read(sp.first.fd, &i, sizeof(i)) == sizeof(i), "Read failed");
  SANDBOX_INVARIANT(i == 0x12345678, "i is {:x}, 0x12345678 expected", i);
}
