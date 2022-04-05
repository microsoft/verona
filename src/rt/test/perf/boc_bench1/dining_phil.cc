// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <memory>
#include <test/harness.h>
#include <test/when.h>

// Command line parametes
size_t HUNGER = 500;
size_t NUM_PHILOSOPHERS = 50;
size_t NUM_TABLES = 1;
bool OPTIMAL_ORDER = false;
size_t WORK_USEC = 1000;

struct Fork : public VCown<Fork>
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

Fork** forks;

void setup_forks()
{
  for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
  {
    forks[i] = new Fork;
    Cown::acquire(forks[i]);
  }
}

Fork* get_left(size_t index)
{
  return forks[index];
}

Fork* get_right(size_t index)
{
  // This code handles tables by splitting the index into the table and then
  // the philosopher index.  It wraps the fork around for the last philosoher
  // on each table.
  size_t table_size = NUM_PHILOSOPHERS / NUM_TABLES;
  size_t table = index / table_size;
  size_t next_fork = (index + 1) % table_size;
  return forks[(table * table_size) + next_fork];
}

struct Philosopher
{
  Fork* left; // Has a reference count on the forks
  Fork* right; // Has a reference count on the forks
  size_t hunger;

  Philosopher(size_t index)
  : left(get_left(index)), right(get_right(index)), hunger(HUNGER)
  {}

  static void eat(std::unique_ptr<Philosopher>&& phil)
  {
    if (phil->hunger > 0)
    {
      when(phil->left, phil->right)
        << [phil = std::move(phil)](Fork* f1, Fork* f2) mutable {
             f1->use();
             f2->use();
             busy_loop(WORK_USEC); // Ponder
             phil->hunger--;
             eat(std::move(phil));
           };
    }
  }

  ~Philosopher()
  {
    Cown::release(ThreadAlloc::get(), left);
    Cown::release(ThreadAlloc::get(), right);
  }
};

void test_body()
{
  if (OPTIMAL_ORDER)
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i += 2)
    {
      auto phil = std::make_unique<Philosopher>(i);
      Philosopher::eat(std::move(phil));
    }
    for (size_t i = 1; i < NUM_PHILOSOPHERS; i += 2)
    {
      auto phil = std::make_unique<Philosopher>(i);
      Philosopher::eat(std::move(phil));
    }
  }
  else
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
    {
      auto phil = std::make_unique<Philosopher>(i);
      Philosopher::eat(std::move(phil));
    }
  }
}

void test1()
{
  setup_forks();
  // Hold no forks during the initial schedule.
  verona::rt::schedule_lambda(test_body);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  HUNGER = harness.opt.is<size_t>("--hunger", HUNGER);
  NUM_PHILOSOPHERS =
    harness.opt.is<size_t>("--num_philosophers", NUM_PHILOSOPHERS);
  NUM_TABLES = harness.opt.is<size_t>("--num_tables", NUM_TABLES);
  OPTIMAL_ORDER = harness.opt.has("--optimal_order");

  if ((NUM_PHILOSOPHERS % NUM_TABLES) != 0)
  {
    std::cerr << "--num_philosophers must be a multiple of --num_tables"
              << std::endl;
  }

  if (NUM_PHILOSOPHERS < 2)
  {
    std::cerr << "--num_philosophers must be at least 2" << std::endl;
    return 1;
  }

  forks = new Fork*[NUM_PHILOSOPHERS * NUM_TABLES];

  harness.run(test1);

  delete[] forks;
}
