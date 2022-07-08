// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <cpp/when.h>
#include <memory>
#include <test/harness.h>

using namespace verona::cpp;

struct Fork
{
  size_t uses{0};

  void use()
  {
    ++uses;
  }

  ~Fork()
  {
    check((HUNGER * 2) == uses);
  }
};

cown_ptr<Fork> get_left(std::vector<cown_ptr<Fork>>& forks, size_t index)
{
  return forks[index];
}

cown_ptr<Fork> get_right(std::vector<cown_ptr<Fork>>& forks, size_t index)
{
  return forks[(index + 1) % NUM_PHILOSOPHERS];
}

struct Philosopher
{
  cown_ptr<Fork> left;
  cown_ptr<Fork> right;
  size_t hunger;

  Philosopher(std::vector<cown_ptr<Fork>>& forks, size_t index)
  : left(get_left(forks, index)), right(get_right(forks, index)), hunger(HUNGER)
  {}

  static void eat(std::unique_ptr<Philosopher> phil)
  {
    if (phil->hunger > 0)
    {
      when(phil->left, phil->right)
        << [phil = std::move(phil)](
             acquired_cown<Fork> f1, acquired_cown<Fork> f2) mutable {
             f1->use();
             f2->use();
             busy_loop(WORK_USEC);
             phil->hunger--;
             eat(std::move(phil));
           };
    }
  }
};

void test_body()
{
  std::vector<cown_ptr<Fork>> forks;

  for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
  {
    forks.push_back(make_cown<Fork>());
  }

  if (OPTIMAL_ORDER)
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i += 2)
    {
      auto phil = std::make_unique<Philosopher>(forks, i);
      Philosopher::eat(std::move(phil));
    }
    for (size_t i = 1; i < NUM_PHILOSOPHERS; i += 2)
    {
      auto phil = std::make_unique<Philosopher>(forks, i);
      Philosopher::eat(std::move(phil));
    }
  }
  else
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
    {
      auto phil = std::make_unique<Philosopher>(forks, i);
      Philosopher::eat(std::move(phil));
    }
  }
}

void test1()
{
  when() << test_body;
}

int verona_main(SystematicTestHarness& harness)
{
  harness.run(test1);

  return 0;
}