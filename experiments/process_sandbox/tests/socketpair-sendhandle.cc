// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "process_sandbox/helpers.h"
#include "process_sandbox/platform/platform.h"

using namespace sandbox::platform;

int main(void)
{
  auto sp = SocketPair::create();
  auto sp2 = SocketPair::create();
  int i = 42;
  // Check sending succeeds
  SANDBOX_INVARIANT(sp.first.send(&i, sizeof(i)), "Send failed");
  i = 12;
  // Check receiving succeeds and gives us the value we wanted
  SANDBOX_INVARIANT(sp.second.receive(&i, sizeof(i)), "Receive failed");
  SANDBOX_INVARIANT(i == 42, "{} != 42", i);
  SANDBOX_INVARIANT(sp.first.send(&i, sizeof(i)), "Send failed");
  i = 12;
  // Check that we can do the receive that might receive a handle even if no
  // handle was sent and still receive the data portion correctly.
  Handle h;
  SANDBOX_INVARIANT(sp.second.receive(&i, sizeof(i), h), "Receive failed");
  // We didn't receive a handle, so ensure that it doesn't look like we did.
  SANDBOX_INVARIANT(!h.is_valid(), "Valid handle received unexpectedly");
  // But we did get the data.
  SANDBOX_INVARIANT(i == 42, "{} != 42", i);
  i = 0x12345678;
  // Send the receive end of a socket pair.
  h = std::move(sp2.second);
  SANDBOX_INVARIANT(sp.first.send(&i, sizeof(i), h), "Sending a socket failed");
  SANDBOX_INVARIANT(
    sp.second.receive(&i, sizeof(i), h), "Receiving a socket failed");
  SANDBOX_INVARIANT(h.is_valid(), "Failed to receive a valid handle");
  // Check that the received socket is really the same one that we sent by
  // sending something to it and checking that we can receive.
  SANDBOX_INVARIANT(
    sp2.first.send(&i, sizeof(i)), "Sending over received socket failed");
  i = 12;
  SocketPair::Socket newsock;
  newsock.reset(h.take());
  SANDBOX_INVARIANT(
    newsock.receive(&i, sizeof(i)), "Receiving over received socket failed");
  SANDBOX_INVARIANT(i == 0x12345678, "Received value {:x} != 0x12345678", i);
}
