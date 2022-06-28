#include "../../../test/harness.h"

#include <cpp/when.h>
#include <string>
#include <unordered_map>

using namespace verona::rt;

const uint32_t GENERATOR_COUNT = 1;
const uint64_t ACCOUNTS_COUNT = 1000;

struct Checking
{
  uint64_t balance;
};

struct Savings
{
  uint64_t balance;
};

struct Account
{
  cown_ptr<Checking> checking;
  cown_ptr<Savings> savings;
};

// Balance transaction: Return sum of savings and checking
void balance(
  std::unordered_map<std::string, Account>& accounts, std::string user_id)
{
  auto account = accounts.at(user_id);
  // Return a promise
  auto pp = Promise<uint64_t>::create_promise();

  when(account.checking, account.savings) <<
    [wp = std::move(pp.second)](
      acquired_cown<Checking> ch_acq, acquired_cown<Savings> sa_acq) mutable {
      Promise<uint64_t>::fulfill(
        std::move(wp), ch_acq->balance + sa_acq->balance);
      std::cout << "Balance accounts";
    };

  pp.first.then([](std::variant<uint64_t, Promise<uint64_t>::PromiseErr> val) {
    if (std::holds_alternative<uint64_t>(val))
    {
      auto v = std::get<uint64_t>(std::move(val));
      std::cout << "Balance result " << v << std::endl;
    }
    else
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
  uint64_t amount)
{
  auto account = accounts.at(user_id);
  when(account.checking) << [=](acquired_cown<Checking> ch_acq) {
    ch_acq->balance += amount;
    std::cout << "deposit checking";
  };
}

// TransactSavings: Add or remove from the savings account
void transact_savings(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id,
  int64_t amount)
{
  auto account = accounts.at(user_id);
  when(account.savings) << [=](acquired_cown<Savings> sa_acq) {
    if ((amount < 0) && (sa_acq->balance < static_cast<uint64_t>(-1 * amount)))
      return;
    sa_acq->balance += amount;
    std::cout << "transact savings";
  };
}

// WriteCheck
void write_check(
  std::unordered_map<std::string, Account>& accounts,
  std::string user_id,
  uint64_t amount)
{
  auto account = accounts.at(user_id);
  when(account.savings, account.checking)
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
  auto account1 = accounts.at(user_id1);
  auto account2 = accounts.at(user_id2);

  when(account1.savings, account1.checking, account2.checking)
    << [=](
         acquired_cown<Savings> sa_acq1,
         acquired_cown<Checking>(ch_acq1),
         acquired_cown<Checking> ch_acq2) {
         ch_acq2->balance += (sa_acq1->balance + ch_acq1->balance);
         sa_acq1->balance = 0;
         ch_acq1->balance = 0;
       };
}

struct Generator
{
  static void generate_tx(acquired_cown<Generator>& g_acq)
  {
    // Business logic

    // Reschedule
    when(g_acq.cown()) <<
      [](acquired_cown<Generator> g_acq_new) { generate_tx(g_acq_new); };
  }
};

void experiment_init()
{
  // Setup the accounts
  auto* accounts = new std::unordered_map<std::string, Account>();

  for (uint64_t i = 0; i < ACCOUNTS_COUNT; i++)
  {
    auto s = make_cown<Savings>();
    auto c = make_cown<Checking>();

    accounts->emplace(std::make_pair(std::to_string(i), Account{c, s}));
  }

  // Setup the generators
  for (uint32_t i = 0; i < GENERATOR_COUNT; i++)
  {
    auto g = make_cown<Generator>();
    when(g) << [](acquired_cown<Generator> g_acq_new) {
      Generator::generate_tx(g_acq_new);
    };
  }
}

void smallbank_body()
{
  when() << []() { std::cout << "Hello smallbank\n"; };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(smallbank_body);
}
