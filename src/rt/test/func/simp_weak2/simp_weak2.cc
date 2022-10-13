// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <cpp/when.h>
#include <test/harness.h>

using namespace verona::cpp;

class Observer;

class Subject
{
public:
  typename cown_ptr<Observer>::weak observer;

  ~Subject()
  {
    Logging::cout() << "Subject destroyed" << Logging::endl;
  }
};

class Observer
{
public:
  cown_ptr<Subject> subject;

  ~Observer()
  {
    Logging::cout() << "Observer destroyed" << Logging::endl;
  }
};

void test_body()
{
  // Create two objects a subject and an observer, then form a cycle with a weak
  // reference to break the cycle. This example tests promoting a weak reference
  // to a strong reference.

  Logging::cout() << "test_build()" << Logging::endl;

  auto subject = make_cown<Subject>();
  auto observer = make_cown<Observer>();

  when(subject) <<
    [observer = observer.get_weak()](acquired_cown<Subject> subject) mutable {
      subject->observer = observer;
    };

  when(observer) <<
    [subject = subject](acquired_cown<Observer> observer) mutable {
      observer->subject = subject;
    };

  when(subject) << [](acquired_cown<Subject> subject) mutable {
    auto observer = subject->observer.promote();
    if (observer)
    {
      when(observer) << [](acquired_cown<Observer>) {
        Logging::cout() << "Observer is alive" << Logging::endl;
      };
    }
    else
    {
      Logging::cout() << "Observer is dead" << Logging::endl;
    }
  };
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test_body);

  return 0;
}
