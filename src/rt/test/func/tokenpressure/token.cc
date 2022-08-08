// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <cpp/token.h>
#include <cpp/when.h>
#include <test/harness.h>

using namespace verona::cpp;

void slow_work(Token t)
{
  busy_loop(1000);
  UNUSED(t);
  printf("!");
}

void fast_work(Token t, cown_ptr<size_t> slow_cown)
{
  printf(".");
  when () << [t = std::move(t), slow_cown]() mutable {
    busy_loop(5000);
    when(slow_cown) << [t = std::move(t)](auto) mutable { slow_work(std::move(t)); };
  };
}

void generate_loop(Token::Source ts, cown_ptr<size_t> log, cown_ptr<size_t> slow_cown, size_t count = 1000)
{
  std::move(ts).get_token([count, log, slow_cown](Token t, auto ts) {
    // do something
    if (count > 0)
    {
      fast_work(std::move(t), slow_cown);
      generate_loop(std::move(ts), log, slow_cown, count - 1);
    }
  });
}

void test()
{
  auto ts = Token::Source::create(40);
  auto log = make_cown<size_t>((size_t)0);
  auto slow_cown = make_cown<size_t>((size_t)0);


  when() << [ts = std::move(ts), log, slow_cown]() mutable { generate_loop(std::move(ts), log, slow_cown); };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test);

  return 0;
}
