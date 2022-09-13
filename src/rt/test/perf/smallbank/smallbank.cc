// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <cpp/when.h>
#include <string>
#include <unordered_map>

using namespace verona::rt;
using namespace verona::cpp;

enum TX_TYPES
{
  BALANCE = 0,
  DEPOSIT_CKN,
  TRANSACT_SNG,
  WRITE_CHECK,
  AMLGMT,
  TX_COUNT
};

struct Checking
{
  int64_t balance;
};

struct Savings
{
  int64_t balance;
};

struct Account
{
  cown_ptr<Checking> checking;
  cown_ptr<Savings> savings;
};

using Accounts = std::unordered_map<std::string, Account>;

// Balance transaction: Return sum of savings and checking
void balance(
  std::unordered_map<std::string, Account>& accounts, std::string user_id)
{
  auto account = accounts.find(user_id);
  if (account == accounts.end())
    return;

  // Return a promise
  auto pp = Promise<int64_t>::create_promise();

  when(account->second.checking, account->second.savings) <<
    [wp = std::move(pp.second)](
      acquired_cown<Checking> ch_acq, acquired_cown<Savings> sa_acq) mutable {
      Promise<int64_t>::fulfill(
        std::move(wp), ch_acq->balance + sa_acq->balance);
    };

  pp.first.then([](std::variant<int64_t, Promise<int64_t>::PromiseErr> val) {
    if (!std::holds_alternative<int64_t>(val))
    {
      Logging::cout() << "Got promise error" << std::endl;
      abort();
    }
  });
}

// DepositChecking: Add to the checking account
void deposit_checking(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id,
  int64_t amount)
{
  auto account = accounts.find(user_id);
  if (account == accounts.end())
    return;

  when(account->second.checking)
    << [=](acquired_cown<Checking> ch_acq) { ch_acq->balance += amount; };
}

// TransactSavings: Add or remove from the savings account
void transact_savings(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id,
  int64_t amount)
{
  auto account = accounts.find(user_id);
  if (account == accounts.end())
    return;

  when(account->second.savings) << [=](acquired_cown<Savings> sa_acq) {
    if ((amount < 0) && (sa_acq->balance < (-1 * amount)))
      return;
    sa_acq->balance += amount;
  };
}

// WriteCheck
void write_check(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id,
  int64_t amount)
{
  auto account = accounts.find(user_id);
  if (account == accounts.end())
    return;

  when(account->second.savings, account->second.checking)
    << [=](acquired_cown<Savings> sa_acq, acquired_cown<Checking> ch_acq) {
         if (amount < (ch_acq->balance + sa_acq->balance))
           ch_acq->balance -= (amount + 1);
         else
           ch_acq->balance -= amount;
       };
}

// Amalgamate: Move all funds from account 1 to the checking account 2
void amalgamate(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id1,
  std::string user_id2)
{
  auto account1 = accounts.find(user_id1);
  auto account2 = accounts.find(user_id2);
  if ((account1 == accounts.end()) || (account2 == accounts.end()))
    return;

  when(
    account1->second.savings,
    account1->second.checking,
    account2->second.checking)
    << [=](
         acquired_cown<Savings> sa_acq1,
         acquired_cown<Checking> ch_acq1,
         acquired_cown<Checking> ch_acq2) {
         ch_acq2->balance += (sa_acq1->balance + ch_acq1->balance);
         sa_acq1->balance = 0;
         ch_acq1->balance = 0;
       };
}

struct Generator
{
private:
  std::shared_ptr<Accounts> accounts;
  uint32_t tx_count;
  std::chrono::time_point<std::chrono::system_clock> start;

  void print_stats()
  {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Generator: " << std::hex << this << ": Dispatched "
              << tx_count / duration.count() << " tx/s\n";
    fflush(stdout);
  }

public:
  Generator(std::shared_ptr<Accounts> accounts_)
  : accounts(accounts_), tx_count(0), start(std::chrono::system_clock::now())
  {}
  static void generate_tx(acquired_cown<Generator>& g_acq)
  {
    // Business logic
    for (uint32_t i = 0; i < TX_BATCH; i++)
    {
      uint8_t txn_type = (uint8_t)rand() % TX_COUNT;
      uint64_t acc1 =
        static_cast<uint64_t>(rand()) % (ACCOUNTS_COUNT + ACCOUNT_EXTRA);

      switch (txn_type)
      {
        case BALANCE:
          balance(*(g_acq->accounts), std::to_string(acc1));
          break;
        case DEPOSIT_CKN:
          deposit_checking(*(g_acq->accounts), std::to_string(acc1), rand());
          break;
        case TRANSACT_SNG:
          transact_savings(*(g_acq->accounts), std::to_string(acc1), rand());
          break;
        case WRITE_CHECK:
          write_check(*(g_acq->accounts), std::to_string(acc1), rand());
          break;
        case AMLGMT:
          uint64_t acc2 =
            static_cast<uint64_t>(rand()) % (ACCOUNTS_COUNT + ACCOUNT_EXTRA);
          while (acc2 == acc1)
            acc2 =
              static_cast<uint64_t>(rand()) % (ACCOUNTS_COUNT + ACCOUNT_EXTRA);
          amalgamate(
            *(g_acq->accounts), std::to_string(acc1), std::to_string(acc2));
          break;
      }
    }
    g_acq->tx_count += TX_BATCH;

    // Reschedule
    if (g_acq->tx_count < PER_GEN_TX_COUNT)
      when(g_acq.cown()) <<
        [](acquired_cown<Generator> g_acq_new) { generate_tx(g_acq_new); };
    else
      g_acq->print_stats();
  }
};

void experiment_init()
{
  // Setup the accounts
  auto accounts = std::make_shared<Accounts>();

  for (uint64_t i = 0; i < ACCOUNTS_COUNT; i++)
  {
    auto s = make_cown<Savings>();
    auto c = make_cown<Checking>();

    accounts->emplace(std::make_pair(std::to_string(i), Account{c, s}));
  }

  // Setup the generators
  for (uint32_t i = 0; i < GENERATOR_COUNT; i++)
  {
    auto g = make_cown<Generator>(accounts);
    when(g) << [](acquired_cown<Generator> g_acq_new) {
      Generator::generate_tx(g_acq_new);
    };
  }
}

void smallbank_body()
{
  experiment_init();
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  if (!process_args(harness))
  {
    return -1;
  }

  harness.run(smallbank_body);
}
