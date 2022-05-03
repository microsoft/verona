// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <cpp/when.h>
#include <ds/scramble.h>
#include <memory>
#include <test/harness.h>

/**
 * This file implements an example of a set of Workers
 * creating transactions.  The transactions use Accounts,
 * and finally, all the transactions are logged.
 *
 * There is a busy loop in the transactions.
 */

struct Account
{
  int64_t balance;
  int64_t overdraft;
  size_t id;

  Account(int64_t balance, int64_t overdraft, size_t id)
  : balance(balance), overdraft(overdraft), id(id)
  {}

  ~Account()
  {
    Logging::cout() << "Account " << id << " destroyed" << Logging::endl;
  }
};

using Accounts = std::vector<cown_ptr<Account>>;

struct Worker
{
  std::shared_ptr<Accounts> accounts;

  xoroshiro::p128r64 rand;

  Worker(std::shared_ptr<Accounts> accounts, size_t seed) : accounts(accounts)
  {
    rand.set_state(seed + 1);
  }

  ~Worker()
  {
    Logging::cout() << "Worker destroyed" << Logging::endl;
  }
};

struct Log
{};

void log(cown_ptr<std::unique_ptr<Log>> log, std::string)
{
  when(log) << [=](auto) { /*std::cout << msg << std::endl;*/ };
}

void bank_job(
  acquired_cown<Worker>& worker,
  cown_ptr<std::unique_ptr<Log>> l,
  size_t repeats)
{
  // Select two accounts at random.
  auto from_idx = worker->rand.next() % NUM_ACCOUNTS;
  auto to_idx = from_idx;

  // We don't want to use the same account.
  while (to_idx == from_idx)
    to_idx = worker->rand.next() % NUM_ACCOUNTS;

  // Schedule a transfer
  int64_t amount = 100;
  Accounts& accounts = *worker->accounts;
  when(accounts[from_idx], accounts[to_idx])
    << [=](acquired_cown<Account> from, acquired_cown<Account> to) {
         // Note that we could do
         //  (auto from, auto to) {
         // We give explicit types for clarity.
         busy_loop(WORK_USEC);

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

  if (repeats > 0)
  {
    // Reschedule bank_job
    // bank_job(worker, l, repeats - 1);
    when(worker.cown()) <<
      [=](acquired_cown<Worker> worker) { bank_job(worker, l, repeats - 1); };
  }
}

void test_body()
{
  Logging::cout() << "test_body()" << Logging::endl;

  auto log = make_cown<std::unique_ptr<Log>>(std::make_unique<Log>());

  // We share accounts across all the workers, use C++
  // memory management to collect after the Workers finish.
  auto accounts = std::make_shared<Accounts>();

  size_t ids = 0;
  for (size_t i = 0; i < NUM_ACCOUNTS; i++)
  {
    Logging::cout() << "Account " << i << Logging::endl;
    accounts->push_back(make_cown<Account>(100, 100, ids++));
  }

  for (size_t j = 0; j < NUM_WORKERS; j++)
  {
    Logging::cout() << "Worker " << j << Logging::endl;
    when(make_cown<Worker>(accounts, j + 1)) << [=](acquired_cown<Worker> w) {
      bank_job(w, log, TRANSACTIONS / NUM_WORKERS);
    };
  }
}

int verona_main(SystematicTestHarness& harness)
{
  harness.run(test_body);

  return 0;
}
