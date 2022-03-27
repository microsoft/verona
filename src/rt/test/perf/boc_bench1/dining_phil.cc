// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ds/scramble.h>
#include <memory>
#include <test/harness.h>

/**
 * This file implements the following program:
 *
class Fork {
  create(): cown[Fork] & imm { cown.create( new Fork) }
  use(self: mut) { }
}

class Philosopher
{
  left: cown[Fork] & imm;
  right: cown[Fork] & imm;

  create(hunger: U64 & imm, left: cown[Fork] & imm, right: cown[Fork] & imm) {
    var result = new Philosopher;
    result.left = left;
    result.right = right;
    result.eat(hunger)
  }

  eat(self: iso, hunger: U64 & imm) {
    if hunger > 0 {
      when (var l = self.left, var r = self.right) {
        l.use(); r.use();
      };
      self.eat(hunger - 1)
    }
  }
}

class Main {

  main() {

    var fork0 = Fork.create();
    ...
    var forkn = Fork.create();

    when (var _0 = fork0, ..., var _4 = forkn) {

      Philosopher.create(w, fork0, fork1);
      Philosopher.create(w, fork1, fork2);
      ...
      Philosopher.create(w, forkn, fork0);
    }
  }
}
 */

// How many uses each fork should have.
size_t HUNGER = 500;
size_t NUM_PHILOSOPHERS = 50;
size_t NUM_TABLES = 100;

bool spawn_all = false;

struct Fork : public VCown<Fork>
{
  size_t uses{0};

  void use()
  {
    ++uses;
    //std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }

  ~Fork()
  {
    check((HUNGER * 2) == uses);
  }
};

struct Philosopher
{
  Fork* forks[2]; // Has a reference count on the forks

  /**
   * Transfer a reference count to the forks to the constructor.
   * TODO: Add C++ smart point type to make this clear.
   */
  Philosopher(Fork* left, Fork* right)
  {
    forks[0] = left;
    forks[1] = right;
  }

  static void eat(std::unique_ptr<Philosopher>&& phil, size_t hunger)
  {
    if (hunger > 0)
    {
      if (spawn_all)
      {
        verona::rt::schedule_lambda(
          2, (Cown**)phil->forks, [fork1 = phil->forks[0], fork2 = phil->forks[1]]() mutable {
            fork1->use();
            fork2->use();
          });
        eat(std::move(phil), hunger - 1);
      }
      else
      {
        verona::rt::schedule_lambda(
          2, (Cown**)phil->forks, [phil = std::move(phil), hunger]() mutable {
            phil->forks[0]->use();
            phil->forks[1]->use();
            eat(std::move(phil), hunger - 1);
          });
      }
    }
  }

  ~Philosopher()
  {
    Cown::release(ThreadAlloc::get(), forks[0]);
    Cown::release(ThreadAlloc::get(), forks[1]);
  }
};

Fork** forks;

void setup_forks()
{
  // Reset forks use count
  for (size_t i = 0; i < NUM_PHILOSOPHERS * NUM_TABLES; i++)
  {
    forks[i] = new Fork;
    Cown::acquire(forks[i]);
  }
}

void test_body()
{
  for (size_t j = 0; j < NUM_TABLES; j++)
  {
    size_t offset = j * NUM_PHILOSOPHERS;
    // Schedule all the eat messages.
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
    {
        std::unique_ptr<Philosopher> phil = std::make_unique<Philosopher>(
        forks[i + offset], forks[offset + ((i + 1) % NUM_PHILOSOPHERS)]);
        Philosopher::eat(std::move(phil), HUNGER);
    }
  }
  printf("Finished scheduling all eat messages\n");
}

void test1()
{
  spawn_all = false;
  setup_forks();
  // Hold no forks during the initial schedule.
  verona::rt::schedule_lambda(test_body);
}

void test2()
{
  setup_forks();
  spawn_all = true;
  // Hold all the forks during the initial schedule.
  verona::rt::schedule_lambda(NUM_PHILOSOPHERS, (Cown**)forks, test_body);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  size_t test_no = harness.opt.is<size_t>("--test_no", 0);
  HUNGER = harness.opt.is<size_t>("--hunger", HUNGER);
  NUM_PHILOSOPHERS =
    harness.opt.is<size_t>("--num_philosophers", NUM_PHILOSOPHERS);
  NUM_TABLES = harness.opt.is<size_t>("--num_tables", NUM_TABLES);

  if (NUM_PHILOSOPHERS < 2)
  {
    std::cerr << "--num_philosophers must be at least 2" << std::endl;
    return 1;
  }

  forks = new Fork*[NUM_PHILOSOPHERS * NUM_TABLES];

  if (test_no == 1)
    harness.run(test1);
  else if (test_no == 2)
    harness.run(test2);
  else
  {
    std::cerr << "--test_no must be 1 or 2" << std::endl;
    return 1;
  }

  delete forks;
}