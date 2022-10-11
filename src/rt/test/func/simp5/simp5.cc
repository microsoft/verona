// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <cpp/when.h>
#include <test/harness.h>

class Body
{
public:
  ~Body()
  {
    Logging::cout() << "Body destroyed" << Logging::endl;
  }
};

using namespace verona::cpp;

void test_body()
{
  Logging::cout() << "test_body()" << Logging::endl;

  auto log1 = make_cown<Body>();
  auto log2 = make_cown<Body>();
  auto log3 = make_cown<Body>();

  when(log1, log2) <<
    [=](auto, auto) { Logging::cout() << "log1" << Logging::endl; };
  when(log2, log3) <<
    [=](auto, auto) { Logging::cout() << "log2" << Logging::endl; };
  when(log1, log3) <<
    [=](auto, auto) { Logging::cout() << "log3" << Logging::endl; };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test_body);

  return 0;
}
