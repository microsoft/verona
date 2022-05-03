#include "args.h"
#include <test/harness.h>

int verona_main(SystematicTestHarness& harness);


int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  // This test is meant to be run on a single core.
  Logging::cout() << "Running with a single core" << std::endl; 
  harness.cores = 1;
  if (!process_args(harness))
    return 1;

  return verona_main(harness);
}

