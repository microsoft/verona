// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ds/scramble.h>

// Simple test that perm is not preserving orders.
int main()
{
  xoroshiro::p128r32 r(2);
  bool failed = false;
  for (size_t i = 0; i < 100; i++)
  {
    verona::Scramble perm;
    perm.setup(r);

    verona::Scramble perm2;
    perm2.setup(r);

    size_t count = 0;

    for (uint8_t* p = nullptr; p < (void*)1000; p++)
    {
      if (perm(p, p + 1) == perm2(p, p + 1))
        count++;
    }

    if (count > 550 || count < 450)
    {
      failed = true;
      std::cout << "Count same: " << count << std::endl;
    }
  }
  if (failed)
    return 1;
  else
    return 0;
}