// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>
#include <map>
#include <sstream>
#include <variant>

namespace verona::parser
{
  struct Subtype
  {
    struct Bounds
    {
      List<Type> lower;
      List<Type> upper;
    };

    using State = std::variant<bool, std::stringstream>;
    using Cache = std::map<std::pair<Node<Type>, Node<Type>>, State>;
    using BoundsMap = std::map<Node<Type>, Bounds, std::owner_less<>>;

    Cache checked;
    BoundsMap bounds;
    std::ostream out;

    Location name_apply;
    Cache::iterator current;
    bool show = true;

    struct NoShow
    {
      Subtype& s;
      bool prev;

      NoShow(Subtype* sub) : s(*sub), prev(s.show)
      {
        s.show = false;
      }

      ~NoShow()
      {
        s.show = prev;
      }
    };

    Subtype() : out(std::cerr.rdbuf()) {}

    void set_error(std::ostream& s)
    {
      out.rdbuf(s.rdbuf());
    }

    std::ostream& error()
    {
      if (show)
      {
        current->second = false;
        return out << "--------" << std::endl;
      }

      current->second = std::stringstream();
      return std::get<std::stringstream>(current->second);
    }

    void result(bool ok)
    {
      auto r = std::get_if<bool>(&current->second);

      if (r)
        *r = ok;
    }

    bool operator()(Node<Type> lhs, Node<Type> rhs);
    bool constraint(Node<Type>& lhs, Node<Type>& rhs);

    void t_sub_t(Node<Type>& lhs, Node<Type>& rhs);

    void infer_sub_t(Node<Type>& lhs, Node<Type>& rhs);
    void union_sub_t(Node<Type>& lhs, Node<Type>& rhs);
    void isect_sub_t(Node<Type>& lhs, Node<Type>& rhs);

    void t_sub_infer(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_union(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_throw(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_isect(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_tuple(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_function(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_typeref(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_class(Node<Type>& lhs, Node<Type>& rhs);
    void t_sub_iface(Node<Type>& lhs, Node<Type>& rhs);

    void sub_same(Node<Type>& lhs, Node<Type>& rhs);

    bool returnval(Cache::iterator& it);
    void kinderror(Node<Type>& lhs, Node<Type>& rhs);
    void unexpected();
  };
}
