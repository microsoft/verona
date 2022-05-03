// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <test/harness.h>

int verona_main(SystematicTestHarness& harness);
int pthread_main();

/*
 * The pthread version will not restrict the number of cores as it creates a
 * thread per philosopher. To correct for this use taskset --cpu-list 0-3
 * ./perf_con-boc_bench1 --pthread --num_philosophers=100 to run with 100
 * threads on the first 4 cores.
 */

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
