#include <test/harness.h>

void smallbank_body()
{
  schedule_lambda([](){ std::cout << "Hello smallbank\n"; });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(smallbank_body);
}
