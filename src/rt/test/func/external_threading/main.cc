// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#define VERONA_EXTERNAL_THREADING
#include <iterator>
#include <test/harness.h>
#include <verona.h>

constexpr const char* HARNESS_ARGV[] = {"binary", "--cores", "1"};

void test_cown()
{
  schedule_lambda([] { Logging::cout() << "Executed" << Logging::endl; });
}

/**
 * This test if only a build test to confirm that the template of
 * verona_external_threading.h matches the requirements of the Verona run-time.
 * Instantiate SystematicTestHarness with 1 core and simple Cown usage to ensure
 * that templates are instantiated.
 */
int main()
{
  Logging::enable_logging();
  SystematicTestHarness harness(
    static_cast<int>(std::size(HARNESS_ARGV)), HARNESS_ARGV);

  harness.run(test_cown);

  return 0;
}
