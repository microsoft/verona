#include <test/harness.h>
#include <cpp/when.h>

#include <unordered_map>

using namespace verona::rt;

struct Checking// : public VCown<Checking>
{
  uint64_t balance;
};

struct Savings// : public VCown<Savings>
{
  uint64_t balance;
};

struct Account
{
  cown_ptr<Checking> checking;
  cown_ptr<Savings> savings;
};

void balance(std::unordered_map<std::string, Account *> &accounts, std::string user_id)
{
  auto *account = accounts.at(user_id);
  when(account->checking, account->savings) << [](acquired_cown<Checking> ch, acquired_cown<Savings> sa) { std::cout << "hello with accounts\n"; };
}

void smallbank_body()
{
  when() << [](){ std::cout << "Hello smallbank\n"; };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(smallbank_body);
}
