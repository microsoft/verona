// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ds/scramble.h>
#include <random>
#include <test/harness.h>

struct Fork : public VCown<Fork>
{
  size_t id;
  size_t uses_expected;
  size_t uses;

  Fork(size_t id) : id(id), uses_expected(0), uses(0){};

  ~Fork()
  {
    assert(uses_expected == uses);
  }
};

struct Ping : public VAction<Ping>
{
  void f() {}
};

/**
 * This Message holds on to the only reference to a Cown, that it
 * will "Ping" once it is delivered.  This is used to find missing
 * scans of messages.  If a message is not scanned, the Cown c
 * will be reclaimable, once this message is delivered it will be
 * deallocated.
 **/
struct KeepAlive : public VAction<KeepAlive>
{
  Cown* c;

  KeepAlive()
  {
    c = new Fork(999);
    Cown::schedule<Ping>(c);
  }

  void trace(ObjectStack& fields) const
  {
    fields.push(c);
  }

  void f()
  {
    Cown::schedule<Ping, YesTransfer>(c);
  }
};

struct Philosopher : public VCown<Philosopher>
{
  size_t id;
  std::vector<Cown*> forks;
  size_t to_eat;

  Philosopher(size_t id_, std::vector<Cown*> forks_, size_t to_eat_)
  : id(id_), forks(forks_), to_eat(to_eat_)
  {}

  void trace(ObjectStack& fields) const
  {
    for (auto f : forks)
    {
      fields.push(f);
    }
  }
};

void eat_send(Philosopher* p);

struct Ponder : public VAction<Ponder>
{
  Philosopher* p;

  Ponder(Philosopher* p) : p(p) {}

  void f()
  {
    Systematic::cout() << "Philosopher " << p->id << " " << p << " pondering "
                       << p->to_eat << std::endl;
    eat_send(p);
    Scheduler::want_ld();
  }
};

struct Eat : public VAction<Eat>
{
  Philosopher* eater;

  void f()
  {
    Systematic::cout() << "Philosopher " << eater->id << " " << eater
                       << " eating (" << this << ")" << std::endl;
    for (auto f : eater->forks)
    {
      ((Fork*)f)->uses++;
    }

    Cown::schedule<Ponder>(eater, eater);
  }

  Eat(Philosopher* p_) : eater(p_)
  {
    Systematic::cout() << "Eat Message " << this << " for Philosopher "
                       << p_->id << " " << p_ << std::endl;
  }

  void trace(ObjectStack& fields) const
  {
    Systematic::cout() << "Calling custom trace" << std::endl;
    fields.push(eater);
  }
};

void eat_send(Philosopher* p)
{
  if (p->to_eat == 0)
  {
    auto* alloc = ThreadAlloc::get();
    Systematic::cout() << "Releasing Philosopher " << p->id << " " << p
                       << std::endl;
    Cown::release(alloc, p);
    return;
  }

  p->to_eat--;
  Cown::schedule<Eat>(p->forks.size(), p->forks.data(), p);

  Cown::schedule<KeepAlive>(p->forks[0]);
}

void test_dining(
  size_t philosophers,
  size_t hunger,
  size_t fork_count,
  SystematicTestHarness* h)
{
  std::vector<Fork*> forks;
  for (size_t i = 0; i < philosophers; i++)
  {
    auto f = new Fork(i);
    forks.push_back(f);
    Systematic::cout() << "Fork " << i << " " << f << std::endl;
  }

  verona::Scramble scrambler;
  xoroshiro::p128r32 rand(h->current_seed());

  for (size_t i = 0; i < philosophers; i++)
  {
    scrambler.setup(rand);

    std::vector<Cown*> my_forks;

    std::sort(forks.begin(), forks.end(), [&scrambler](Fork*& a, Fork*& b) {
      return scrambler(((Cown*)a)->id(), ((Cown*)b)->id());
    });

    for (size_t j = 0; j < fork_count; j++)
    {
      forks[j]->uses_expected += hunger;
      Cown::acquire(forks[j]);
      my_forks.push_back(forks[j]);
    }

    auto p = new Philosopher(i, my_forks, hunger);
    Cown::schedule<Ponder>(p, p);
    Systematic::cout() << "Philosopher " << i << " " << p << std::endl;
    for (size_t j = 0; j < fork_count; j++)
    {
      Systematic::cout() << "   Fork " << ((Fork*)my_forks[j])->id << " "
                         << my_forks[j] << std::endl;
    }
  }

  for (size_t i = 0; i < philosophers; i++)
  {
    Cown::release(ThreadAlloc::get_noncachable(), forks[i]);
  }
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  size_t phil = harness.opt.is<size_t>("--philosophers", 4);
  std::cout << " --philosophers " << phil << std::endl;
  size_t hunger = harness.opt.is<size_t>("--hunger", 4);
  std::cout << " --hunger " << hunger << std::endl;
  size_t forks = harness.opt.is<size_t>("--forks", 2);
  std::cout << " --forks " << forks << std::endl;

  if (forks > phil)
  {
    phil = forks;
    std::cout << " overriding philosophers as need as many as forks."
              << std::endl;
  }

  harness.run(test_dining, phil, hunger, forks, &harness);

  return 0;
}
