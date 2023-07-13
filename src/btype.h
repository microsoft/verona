// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"
#include "lookup.h"

namespace verona
{
  struct BtypeDef;
  using Btype = std::shared_ptr<BtypeDef>;

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
        if (node->type().in({Type, TypePred}))
        {
          node = node / Type;
        }
        else if (node->type().in(
                   {TypeClassName,
                    TypeTraitName,
                    TypeAliasName,
                    TypeParamName}))
        {
          auto defs = lookup_scopedname(node);

          // This won't be empty in non-testing code.
          if (defs.defs.empty())
            return;

          // Use existing bindings if they haven't been specified here.
          auto& def = defs.defs.front();
          node = def.def;

          for (auto& bind : def.bindings)
            bindings[bind.first] = make(bind.second, b);

          // Check for cycles.
          if (set.contains(node))
            return;
        }
        else if (node->type() == TypeParam)
        {
          // An unbound typeparam effectively binds to itself.
          set.insert(node);

          auto it = bindings.find(node);
          if (it == bindings.end())
            return;

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

    Btype field(const Token& f)
    {
      return make(node / f, bindings);
    }

    const Token& type() const
    {
      return node->type();
    }

    bool valid_predicate()
    {
      // A predicate is a type that can be used in a where clause. They can be
      // composed of unions and intersections of predicates and type aliases
      // that expand to predicates.
      if (node->type() == TypeSubtype)
      {
        return true;
      }
      else if (node->type().in({TypeUnion, TypeIsect}))
      {
        // Check that all children are valid predicates.
        return std::all_of(node->begin(), node->end(), [&](auto& t) {
          return make(t)->valid_predicate();
        });
      }
      else if (node->type() == TypeAlias)
      {
        return field(Type)->valid_predicate();
      }

      return false;
    }

    bool valid_inherit()
    {
      // A type that can be used in an inherit clause. They can be composed of
      // intersections of classes, traits, and type aliases that expand to
      // valid inherit clauses.
      if (node->type().in({Class, TypeTrait}))
      {
        return true;
      }
      else if (node->type().in({Type, TypeIsect}))
      {
        // Check that all children are valid for code reuse.
        return std::all_of(node->begin(), node->end(), [&](auto& t) {
          return make(t)->valid_inherit();
        });
      }
      else if (node->type() == TypeAlias)
      {
        return field(Type)->valid_inherit();
      }

      return false;
    }

    void str(std::ostream& out, size_t level)
    {
      out << indent(level) << "btype: {" << std::endl
          << indent(level + 1) << "bindings: {" << std::endl;

      for (auto& b : bindings)
      {
        out << indent(level + 2) << "{" << std::endl;
        b.first->str(out, level + 3);
        out << " =" << std::endl;
        b.second->str(out, level + 3);
        out << indent(level + 2) << "}," << std::endl;
      }

      out << indent(level + 1) << "}," << std::endl
          << indent(level + 1) << "node: {" << std::endl;

      if (node->type().in({Class, TypeAlias}))
      {
        out << indent(level + 2) << node->type().str() << " "
            << (node / Ident)->location().view();
      }
      else
      {
        node->str(out, level + 2);
      }

      out << std::endl
          << indent(level + 1) << "}" << std::endl
          << indent(level) << "}" << std::endl;
    }
  };

  inline Btype make_btype(Node t)
  {
    return BtypeDef::make(t, {});
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
