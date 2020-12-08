// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "dispatch.h"

namespace verona::parser
{
  enum class Order
  {
    Pre,
    Post,
  };

  template<typename F, Order order>
  struct Pass
  {
    List<NodeDef> stack;

    Pre(Node<NodeDef>& node)
    {
      stack.push_back(node);
    }

    ~Pre()
    {
      assert(stack.size() == 1);
    }

    Pre<F>& operator<<(Node<NodeDef>& node)
    {
      stack.push_back(node);
      dispatch(*this, node);
      stack.pop_back();
      return *this;
    }

    template<typename T>
    Pre<F> operator<<(List<NodeDef>& nodes)
    {
      for (auto& node : nodes)
        *this << node;

      return *this;
    }

    void operator()() {}

    template<typename T>
    void operator()(T& node)
    {
      if constexpr (order == Pre)
        F(stack, node);

      *this << fields(node);

      if constexpr (order == Post)
        F(stack, node);
    }

    template<>
    void operator()(Token& tok)
    {
      F(stack, tok);
    }

    template<>
    void operator()(Location& loc)
    {
      F(stack, loc);
    }
  };
}
