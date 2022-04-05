// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ds/scramble.h>
#include <memory>
#include <test/harness.h>
#include <test/when.h>

/**
 * This file implements the following program:
 *
 * TODO
 */

size_t OPERATIONS = 100;
size_t NUM_BANKS = 100;

struct Bank : public VCown<Bank>
{
  xoroshiro::p128r64 rand;
};

struct Account : public VCown<Account>
{
  int64_t balance;
  int64_t overdraft;
  size_t id;
};

/**
 *
 */
struct Log : public VCown<Log>
{};

std::array<Account*, 10000> accounts;

void setup_accounts()
{
  size_t ids = 0;
  for (auto& a : accounts)
  {
    a = new Account;
    a->balance = 100;
    a->overdraft = 100;
    a->id = ids++;
  }
}

void log(Log* log, std::string msg)
{
  when(log) << [=](Log*) { std::cout << msg << std::endl; };
}

void bank_job(Bank* bank, Log* l, int repeats)
{
  assert(bank != nullptr);
  for (size_t i = 0; i < OPERATIONS; i++)
  {
    // Select two accounts at random.
    auto from_idx = bank->rand.next() % accounts.size();
    auto to_idx = bank->rand.next() % accounts.size();
    // Runtime currently doesn't deduplicate cown acquisitions, so this leads to
    // deadlock if the two accounts are the same.
    if (to_idx == from_idx)
      continue;
    // Schedule a transfer
    int64_t amount = 100;
    when(accounts[from_idx], accounts[to_idx])
      << [=](Account* from, Account* to) {
           busy_loop(500);

           if ((from->balance + from->overdraft) < amount)
           {
             log(l, "Insufficient funds");
             return;
           }
           else
           {
             from->balance -= amount;
             to->balance += amount;
             log(l, "Success");
           }
         };
  }

  // Reschedule bank_job
  if (repeats > 0)
  {
    when(bank) << [=](Bank* bank) { bank_job(bank, l, repeats - 1); };
  }
  else
  {
    // Tidy up
    verona::rt::Cown::release(ThreadAlloc::get(), bank);
  }
}

void test_body()
{
  Log* log = new Log;
  setup_accounts();
  for (size_t j = 0; j < NUM_BANKS; j++)
  {
    auto b = new Bank;
    // Give each bank a different random seed
    // so they all do different transactions.
    b->rand.set_state(j + 1);

    when(b) << [=](Bank* b) { bank_job(b, log, 10); };
  }
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  // Not correctly doing memory management in this test.
  harness.detect_leaks = false;

  harness.run(test_body);
}
