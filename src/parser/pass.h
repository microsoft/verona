// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "dispatch.h"
#include "fields.h"
#include "rewrite.h"

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
    size_t index;
    bool ok;
    std::ostream out;

    Pass() : index(0), ok(true), out(std::cerr.rdbuf()) {}

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
      if (!stack.empty())
        return stack.back()->location;

      return {};
    }

    text line()
    {
      return text(loc());
    }

    template<typename T = NodeDef>
    Node<T> current()
    {
      if (!stack.empty())
        return std::static_pointer_cast<T>(stack.back());

      return {};
    }

    template<typename T = NodeDef>
    Node<T> parent()
    {
      if (stack.size() > 1)
        return std::static_pointer_cast<T>(stack[stack.size() - 2]);

      return {};
    }

    Ast symbols()
    {
      for (auto it = stack.rbegin(); it != stack.rend(); ++it)
      {
        if ((*it)->symbol_table() != nullptr)
          return *it;
      }

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
      auto prev = index;
      index = 0;

      for (auto& node : nodes)
      {
        *this << node;
        index++;
      }

      index = prev;
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

    bool rewrite(Ast next)
    {
      return parser::rewrite(stack, index, next);
    }
  };
}
