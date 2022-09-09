// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <cpp/when.h>
#include <memory>
#include <test/harness.h>

using namespace verona::cpp;

struct Account
{
  int balance;
  Account(int balance) : balance(balance) {}
};

void test_read_only()
{
  size_t num_accounts = 8;

  std::vector<cown_ptr<Account>> accounts;
  for (size_t i = 0; i < num_accounts; i++)
    accounts.push_back(make_cown<Account>(0));

  cown_ptr<Account> common_account = make_cown<Account>(100);
  when(common_account) <<
    [](acquired_cown<Account> account) { account->balance -= 10; };

  for (size_t i = 0; i < num_accounts; i++)
  {
    when(accounts[i], read(common_account))
      << [](
           acquired_cown<Account> write_account,
           acquired_cown<const Account> ro_account) {
           write_account->balance = ro_account->balance;
         };

    when(read(accounts[i])) << [](acquired_cown<const Account> account) {
      check(account->balance == 90);
    };
  }

  when(common_account) <<
    [](acquired_cown<Account> account) { account->balance += 10; };

  when(read(common_account)) << [](acquired_cown<const Account> account) {
    check(account->balance == 100);
  };
}

void test_read_only_fast_send()
{
  // Test that fast sending to a read cown also reschedules the cown
  cown_ptr<Account> account_one = make_cown<Account>(100);
  cown_ptr<Account> account_two = make_cown<Account>(100);

  when(read(account_one), read(account_two))
    << [](
         acquired_cown<const Account> account_one,
         acquired_cown<const Account> account_two) {
         check(account_one->balance == account_two->balance);
       };

  when(read(account_one), read(account_two))
    << [](
         acquired_cown<const Account> account_one,
         acquired_cown<const Account> account_two) {
         check(account_one->balance == account_two->balance);
       };

  when(account_one, account_two) <<
    [](acquired_cown<Account> account_one, acquired_cown<Account> account_two) {
      check(account_one->balance == account_two->balance);
    };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(test_read_only);
}