#include "args.h"

#include <cpp/when.h>
#include <atomic>
#include <test/harness.h>

std::atomic<size_t> counter = 0;

// Long running behavior that requires behaviors scheduled after it
// to make progress (i.e., not starve due to the core being hogged) in order
// to finish.
// This tests the system monitor's ability to spawn new scheduler threads.
void long_running()
{
  Logging::cout() << "Starting the long running" << std::endl;
  while(counter != 0) {}
  Logging::cout() << "Long running behavior just finished" << std::endl;
}

void short_running()
{
  Logging::cout() << "Running short behavior" << std::endl;
  counter--;
}

void test1()
{
  // Set the counter to the number of short-lived behaviors.
  counter = NUM_SMALL;
  when() << long_running;
  for (size_t i = 0; i < NUM_SMALL; i++)
  {
    when() << short_running;
  }
}


int verona_main(SystematicTestHarness& harness)
{
// This test is unable to complete without the system monitor, which is 
// disabled with systematic testing.
#ifndef USE_SYSTEMATIC_TESTING 
#ifdef USE_SYSTEM_MONITOR
  harness.run(test1); 
#endif
#else
  UNUSED(harness);
#endif
  return 0;
}
