// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"
#include "lookup.h"

namespace verona
{
  struct BtypeDef;
  using Btype = std::shared_ptr<BtypeDef>;
  using Btypes = std::vector<Btype>;

  struct BtypeDef
  {
    Node node;
    NodeMap<Btype> bindings;

    BtypeDef(Node t, NodeMap<Btype> b) : node(t), bindings(b)
    {
      // Keep unwinding until done.
      NodeSet set;

      while (true)
      {
        if (node->in({Type, TypePred}))
        {
          node = node / Type;
        }
        else if (node->in({FQType, FQFunction}))
        {
          auto lookup = resolve_fq(node);

          // This should only happen in test code.
          if (!lookup.def)
            return;

          node = lookup.def;

          // Use existing bindings if they haven't been specified here.
          for (auto& bind : lookup.bindings)
            bindings[bind.first] = make(bind.second, b);

          // Check for cycles.
          if (set.contains(node))
            return;
        }
        else if (node == TypeParam)
        {
          set.insert(node);
          auto it = bindings.find(node);

          // Except in testing, there should always be a binding.
          if (it == bindings.end())
            return;

          // If it's bound to itself, check the next binding.
          if (it->second->type() == TypeParamBind)
          {
            auto bound = it->second;
            for (auto& bind : bound->bindings)
              bindings[bind.first] = bind.second;

            it = bindings.find(node);

            if ((it == bindings.end()) || (it->second->type() == TypeParamBind))
              return;
          }

          *this = *it->second;
        }
        else
        {
          return;
        }
      }
    }

    static Btype make(Node t, NodeMap<Btype> b)
    {
      return std::make_shared<BtypeDef>(t, b);
    }

    Btype make(Node t)
    {
      return make(t, bindings);
    }

    const Token& type() const
    {
      return node->type();
    }

    bool in(const std::initializer_list<Token>& list) const
    {
      return node->in(list);
    }

    void str(std::ostream& out, size_t level)
    {
      out << indent(level) << "btype: {" << std::endl;

      // Print the node.
      out << indent(level + 1) << "node: {" << std::endl;

      if (node->in({Class, TypeAlias, Function}))
      {
        out << indent(level + 2) << node->type().str() << " "
            << (node / Ident)->location().view();
      }
      else
      {
        node->str(out, level + 2);
      }

      out << std::endl << indent(level + 1) << "}," << std::endl;

      // Print the bindings.
      out << indent(level + 1) << "bindings: {" << std::endl;

      for (auto& b : bindings)
      {
        out << indent(level + 2) << "{" << std::endl;
        b.first->str(out, level + 3);
        out << " =" << std::endl;
        b.second->str(out, level + 3);
        out << indent(level + 2) << "}," << std::endl;
      }

      out << indent(level + 1) << "}" << std::endl
          << indent(level) << "}" << std::endl;
    }
  };

  inline Btype make_btype(Node t)
  {
    return BtypeDef::make(t, {});
  }

  inline Btype operator/(Btype& b, const Token& f)
  {
    return b->make(b->node / f);
  }

  inline bool operator==(const Btype& b, const Token& type)
  {
    return b->type() == type;
  }

  inline std::ostream& operator<<(std::ostream& out, const Btype& b)
  {
    b->str(out, 0);
    return out;
  }

  [[gnu::used]] inline void print(const Btype& b)
  {
    std::cout << b;
  }
}
