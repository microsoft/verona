// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include "args.h"

int verona_main(SystematicTestHarness& harness);
int pthread_main();

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  if (!process_args(harness))
  {
    return 1;
  }

  if (harness.opt.has("--pthread"))
  {
    return pthread_main();
  }
  else
  {
    return verona_main(harness);
  }
}
