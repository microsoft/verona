// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "dispatch.h"
#include "fields.h"

#include <iostream>

namespace verona::parser
{
#define AST_PASS \
  void pre(NodeDef& node) {} \
  void post(NodeDef& node) {}

  template<typename F>
  struct Pass
  {
    AstPath stack;
    bool ok;

    Pass() : ok(true) {}

    operator bool() const
    {
      return ok;
    }

    std::ostream& error()
    {
      ok = false;
      return std::cerr;
    }

    Location loc()
    {
      if (stack.size() > 0)
        return stack.back()->location;

      return {};
    }

    text line()
    {
      return text(loc());
    }

    Pass& operator<<(Token& tok)
    {
      // Handle token fields from the node handling functions.
      return *this;
    }

    Pass& operator<<(Location& loc)
    {
      // Handle location fields from the node handling functions.
      return *this;
    }

    Pass& operator<<(Ast node)
    {
      stack.push_back(node);
      dispatch(*this, node);
      stack.pop_back();
      return *this;
    }

    template<typename T>
    Pass& operator<<(List<T>& nodes)
    {
      for (auto& node : nodes)
        *this << node;

      return *this;
    }

    void operator()() {}

    template<typename T>
    void operator()(T& node)
    {
      static_cast<F*>(this)->pre(node);
      *this << fields(node);
      static_cast<F*>(this)->post(node);
    }
  };
}
