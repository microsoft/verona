// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../../../verona.h"

#include <functional>
#include <vector>

namespace backpressure
{
  struct TestBehaviour : public verona::rt::VBehaviour<TestBehaviour>
  {
    std::function<void()> closure;

    TestBehaviour(std::function<void()> closure_) : closure(closure_) {}

    void f()
    {
      closure();
    }
  };

  struct EmptyCown : public verona::rt::VCown<EmptyCown>
  {
    std::vector<Cown*> refs;

    EmptyCown(std::vector<Cown*> refs_ = {}) : refs(refs_)
    {
      for (auto* c : refs)
        verona::rt::Cown::acquire(c);
    }

    ~EmptyCown()
    {
      auto* alloc = snmalloc::ThreadAlloc::get();
      for (auto* c : refs)
        verona::rt::Cown::release(alloc, c);
    }

    void add_ref(Cown* ref)
    {
      refs.push_back(ref);
      Cown::acquire(ref);
    }

    void trace(verona::rt::ObjectStack& s) const
    {
      for (auto* c : refs)
        s.push(c);
    }
  };
}
