// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <cpp/token.h>
#include <cpp/when.h>
#include <test/harness.h>

using namespace verona::cpp;

void slow_work(Token t)
{
  busy_loop(1);
  UNUSED(t);
  printf("!");
}

void fast_work(Token t, cown_ptr<size_t> slow_cown)
{
  printf(".");
  when() << [t = std::move(t), slow_cown]() mutable {
    busy_loop(5);
    when(slow_cown) <<
      [t = std::move(t)](auto) mutable { slow_work(std::move(t)); };
  };
}

void generate_loop(
  Token::Source ts,
  cown_ptr<size_t> log,
  cown_ptr<size_t> slow_cown,
  size_t count = 10000)
{
  size_t i = ts.available_tokens();

  if (i == 0)
  {
    // No tokens available, stop polling and move to waiting.
    std::move(ts).wait_for_token([count, log, slow_cown](auto ts) {
      generate_loop(std::move(ts), log, slow_cown, count);
    });
    return;
  }

  // Calculate if we can complete with the currently available tokens.
  if (count < i)
  {
    i = count;
  }

  // Remove the work we are about to schedule.
  count -= i;

  // Schedule work
  for (; i > 0; i--)
  {
    fast_work(ts.get_token(), slow_cown);
  }

  // If we have more work to do, schedule a continuation.
  if (count != 0)
  {
    when() << [ts = std::move(ts), log, slow_cown, count]() mutable {
      generate_loop(std::move(ts), log, slow_cown, count);
    };
  }
}

void test()
{
  auto ts = Token::Source::create(10);
  auto log = make_cown<size_t>();
  auto slow_cown = make_cown<size_t>();

  when() << [ts = std::move(ts), log, slow_cown]() mutable {
    generate_loop(std::move(ts), log, slow_cown);
  };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test);

  return 0;
}
