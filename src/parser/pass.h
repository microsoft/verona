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
    std::ostream out;

    Pass() : ok(true), out(std::cerr.rdbuf()) {}

    operator bool() const
    {
      return ok;
    }

    void set_error(std::ostream& s)
    {
      out.rdbuf(s.rdbuf());
    }

    std::ostream& error()
    {
      ok = false;
      return out << "--------" << std::endl;
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

    Ast parent()
    {
      if (stack.size() > 1)
        return stack[stack.size() - 2];

      return {};
    }

    Pass& operator<<(Location& loc)
    {
      // Handle location fields from the node handling functions.
      return *this;
    }

    template<typename T>
    Pass& operator<<(Node<T>& node)
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
      auto& check = stack.back();
      static_cast<F*>(this)->pre(node);

      // Don't continue if this node was replaced.
      if (stack.back() != check)
        return;

      *this << fields(node);
      static_cast<F*>(this)->post(node);
    }
  };
}
