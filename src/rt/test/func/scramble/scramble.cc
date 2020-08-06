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
    verona::Scramble s1;
    s1.setup(r);

    verona::Scramble s2;
    s2.setup(r);

    size_t count = 0;

    for (uintptr_t p = 0; p < 1000; p++)
    {
      if ((s1.perm(p) < s1.perm(p + 1)) == (s2.perm(p) < s2.perm(p + 1)))
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
