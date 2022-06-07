// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <test/log.h>
#include <test/opt.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

void test_multimessage(size_t cores)
{
  struct CCown : public VCown<CCown>
  {
    int i = 0;

    ~CCown()
    {
      logger::cout() << "Cown " << (void*)this << " destroyed" << std::endl;
    }
  };

  struct GotMsg : public VBehaviour<GotMsg>
  {
    Cown* a;
    GotMsg(Cown* a) : a(a) {}

    void f()
    {
      logger::cout() << "got message on " << a << std::endl;
    }
  };

  class AddTwo : public VBehaviour<AddTwo>
  {
  private:
    CCown* a;
    CCown* b;

  public:
    AddTwo(CCown* a, CCown* b) : a(a), b(b) {}

    void f()
    {
      logger::cout() << "result = " << (a->i + b->i) << std::endl;
    }

    void trace(ObjectStack& st) const
    {
      st.push(a);
      st.push(b);
    }
  };

  Scheduler& sched = Scheduler::get();
  sched.init(cores);

  auto a1 = new CCown;
  a1->i = 3;
  Cown::schedule<GotMsg>(a1, a1);

  auto a2 = new CCown;
  a2->i = 5;

  // We are transfering our cown references to the message here.
  Cown* dest[2] = {a1, a2};
  Cown::schedule<AddTwo>(2, dest, a1, a2);

  // Show that participating cowns remain alive.
  auto& alloc = ThreadAlloc::get();
  Cown::release(alloc, a1);
  Cown::release(alloc, a2);

  sched.run();
  snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
}

int main(int argc, char** argv)
{
  opt::Opt opt(argc, argv);
  test_multimessage(opt.is<size_t>("--cores", 4));
  return 0;
}
