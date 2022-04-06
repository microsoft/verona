// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <ds/scramble.h>
#include <memory>
#include <test/harness.h>
#include <test/when.h>

/**
 * This file implements the following program:
 *
 * TODO
 */

struct Worker : public VCown<Worker>
{
  xoroshiro::p128r64 rand;
};

struct Account : public VCown<Account>
{
  int64_t balance;
  int64_t overdraft;
  size_t id;
};

struct Log : public VCown<Log>
{};

namespace
{
  Account** accounts;
}

void setup_accounts()
{
  size_t ids = 0;
  for (size_t i = 0; i < NUM_ACCOUNTS; i++)
  {
    accounts[i] = new Account;
    accounts[i]->balance = 100;
    accounts[i]->overdraft = 100;
    accounts[i]->id = ids++;
  }
}

void log(Log* log, std::string)
{
  when(log) << [=](Log*) { /*std::cout << msg << std::endl;*/ };
}

void bank_job(Worker* worker, Log* l, size_t repeats)
{
  // Select two accounts at random.
  auto from_idx = worker->rand.next() % NUM_ACCOUNTS;
  auto to_idx = from_idx;
  // We don't want to use the same account.
  while (to_idx == from_idx)
    to_idx = worker->rand.next() % NUM_ACCOUNTS;

  // Schedule a transfer
  int64_t amount = 100;
  when(accounts[from_idx], accounts[to_idx])
    << [=](Account* from, Account* to) {
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
    //bank_job(worker, l, repeats - 1);
    when(worker) << [=](Worker* worker) { bank_job(worker, l, repeats - 1); };
  }
  else
  {
    // Tidy up
    verona::rt::Cown::release(ThreadAlloc::get(), worker);
  }
}

void test_body()
{
  Log* log = new Log;
  setup_accounts();
  for (size_t j = 0; j < NUM_WORKERS; j++)
  {
    auto w = new Worker;
    // Give each bank a different random seed
    // so they all do different transactions.
    w->rand.set_state(j + 1);

    when(w) << [=](Worker* w) { bank_job(w, log, TRANSACTIONS / NUM_WORKERS); };
  }
}

int verona_main(SystematicTestHarness& harness)
{
  // Not correctly doing memory management in this test.
  harness.detect_leaks = false;

  //NUM_WORKERS = 1;

  accounts = new Account*[NUM_ACCOUNTS];

  harness.run(test_body);

  delete accounts;

  return 0;
}
