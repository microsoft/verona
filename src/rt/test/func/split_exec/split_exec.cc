// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

using namespace std;

struct TestCown : public VCown<TestCown>
{};

template<typename T>
struct foo
{
  size_t count;
  T* items;
};

void split_exec()
{
#ifdef ACQUIRE_ALL
  // foo<int> f;
  // f.count = 42;

  Cown* cowns[2];
  auto& alloc = ThreadAlloc::get();
  cowns[0] = new (alloc) TestCown;
  cowns[1] = new (alloc) TestCown;

  schedule_lambda_many<YesTransfer>(
    LambdaBehaviourPackedArgs([]() { cout << "hello1\n"; }, cowns, 1),
    LambdaBehaviourPackedArgs([]() { cout << "hello2\n"; }, &cowns[1], 1));
#endif
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(split_exec);

  return 0;
}
