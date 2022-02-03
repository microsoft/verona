// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

using namespace std;

void promise_test()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto rp2 = rp;
  auto wp = std::move(pp.second);

  schedule_lambda([wp = std::move(wp)]() mutable {
    Promise<int>::fulfill(std::move(wp), 42);
  });

  rp.then([](std::variant<int, Promise<int>::PromiseErr> val) {
    if (std::holds_alternative<int>(val))
    {
      auto v = std::get<int>(std::move(val));
      Logging::cout() << v << std::endl;
    }
    else
    {
      Logging::cout() << "Got promise error" << std::endl;
      abort();
    }
  });
  rp.then([](std::variant<int, Promise<int>::PromiseErr> val) {
    if (std::holds_alternative<int>(val))
    {
      auto v = std::get<int>(std::move(val));
      Logging::cout() << v << std::endl;
    }
    else
    {
      Logging::cout() << "Got promise error" << std::endl;
      abort();
    }
  });
  rp2.then([](std::variant<int, Promise<int>::PromiseErr> val) {
    if (std::holds_alternative<int>(val))
    {
      auto v = std::get<int>(std::move(val));
      Logging::cout() << v << std::endl;
    }
    else
    {
      Logging::cout() << "Got promise error" << std::endl;
      abort();
    }
  });
}

void promise_no_reader()
{
  auto pp = Promise<int>::create_promise();
  auto wp = std::move(pp.second);

  schedule_lambda([wp = std::move(wp)]() mutable {
    Promise<int>::fulfill(std::move(wp), 42);
  });
}

void promise_no_writer()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);

  rp.then([](std::variant<int, Promise<int>::PromiseErr> val) {
    if (std::holds_alternative<int>(val))
    {
      auto v = std::get<int>(std::move(val));
      Logging::cout() << v << std::endl;
      abort();
    }
    else
    {
      Logging::cout() << "Got promise error" << std::endl;
    }
  });
}

void promise_smart_pointer()
{
  auto pp = Promise<unique_ptr<int>>::create_promise();
  auto rp = std::move(pp.first);
  auto wp = std::move(pp.second);

  schedule_lambda([wp = std::move(wp)]() mutable {
    auto a = make_unique<int>(42);
    Promise<unique_ptr<int>>::fulfill(std::move(wp), std::move(a));
  });

  rp.then(
    [](std::variant<unique_ptr<int>, Promise<unique_ptr<int>>::PromiseErr> a) {
      if (std::holds_alternative<unique_ptr<int>>(a))
      {
        auto p = std::get<unique_ptr<int>>(std::move(a));
        Logging::cout() << *p << std::endl;
      }
      else
      {
        Logging::cout() << "Got promise error" << std::endl;
        abort();
      }
    });
}

void promise_transfer1()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto wp = std::move(pp.second);

  Promise<int>* p = rp.get_promise();
  Logging::cout() << p << std::endl;
  Cown::release(ThreadAlloc::get(), p);
}

void promise_transfer2()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto wp = std::move(pp.second);

  Promise<int>* p1 = rp.get_promise<NoTransfer>();
  Promise<int>* p2 = rp.get_promise<YesTransfer>();

  Logging::cout() << p1 << std::endl;
  Logging::cout() << p2 << std::endl;

  Cown::release(ThreadAlloc::get(), p1);
  auto rp2 = Promise<int>::PromiseR(p2, YesTransfer);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(promise_test);
  harness.run(promise_no_reader);
  harness.run(promise_no_writer);
  harness.run(promise_smart_pointer);
  harness.run(promise_transfer2);

  return 0;
}
