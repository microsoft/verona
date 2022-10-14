// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <cpp/when.h>
#include <test/harness.h>

using namespace verona::cpp;

class Body
{
public:
  typename cown_ptr<Body>::weak self;

  ~Body()
  {
    Logging::cout() << "Body destroyed" << Logging::endl;
  }
};

using namespace verona::cpp;

void test_body()
{
  // Simple example that creates a self loop.

  Logging::cout() << "test_body()" << Logging::endl;

  auto log = make_cown<Body>();

  when(log) << [=](auto log) {
    // Create a self reference, this should not prevent the body from
    // being destroyed.
    log->self = log.cown();
  };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test_body);

  return 0;
}
