// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "rewrite.h"

#include "dispatch.h"
#include "dnf.h"
#include "fields.h"

namespace verona::parser
{
  struct Rewrite
  {
    size_t index;
    Ast prev;
    Ast next;
    bool ok = false;

    bool operator()(size_t index, Ast& prev, Ast& next)
    {
      return false;
    }

    template<typename T>
    bool operator()(T& parent, size_t index, Ast& prev, Ast& next)
    {
      this->index = index;
      this->prev = prev;
      this->next = next;
      ok = false;

      *this << fields(parent);
      return ok;
    }

    Rewrite& operator<<(Location& loc)
    {
      return *this;
    }

    template<typename T>
    Rewrite& operator<<(Node<T>& node)
    {
      if (!ok && (node == prev))
      {
        node = std::static_pointer_cast<T>(next);
        ok = true;
      }

      return *this;
    }

    template<typename T>
    Rewrite& operator<<(Weak<T>& node)
    {
      return *this;
    }

    template<typename T>
    Rewrite& operator<<(List<T>& list)
    {
      if (!ok && (index < list.size()) && (list.at(index) == prev))
      {
        list[index] = std::static_pointer_cast<T>(next);
        ok = true;
      }

      return *this;
    }
  };

  bool rewrite(Ast& parent, size_t index, Ast& prev, Ast next)
  {
    Rewrite rewrite;
    dispatch(rewrite, parent, index, prev, next);
    return rewrite.ok;
  }

  struct Clone
  {
    Substitutions& subs;
    Ast& self;

    Clone(Substitutions& subs, Ast& self) : subs(subs), self(self) {}

    Ast operator()()
    {
      return {};
    }

    Ast operator()(LookupRef& lr)
    {
      // global subs: T -> A | B
      // local subs: U -> T | C
      // rewrite local subs to: U -> A | B | C
      auto clone = std::make_shared<LookupRef>();
      *clone = lr;

      for (auto& sub : clone->subs)
      {
        sub.second =
          std::static_pointer_cast<Type>(dispatch(*this, sub.second));
      }

      return clone;
    }

    Ast operator()(TypeRef& tr)
    {
      // If this is a reference to a TypeParam, and that TypeParam is
      // present in our substitution map, return the substituted type
      // instead.
      if (tr.lookup && (tr.lookup->kind() != Kind::LookupRef))
      {
        auto def = tr.lookup->as<LookupRef>().def.lock();

        if (def && (def->kind() == Kind::TypeParam))
        {
          auto tp = std::static_pointer_cast<TypeParam>(def);
          auto find = subs.find(def);

          if (find != subs.end())
            return find->second;
        }
      }

      auto clone = std::make_shared<TypeRef>();
      *clone = tr;

      *this << fields(*clone);
      return clone;
    }

    Ast operator()(Self& s)
    {
      if (self)
        return self;

      auto clone = std::make_shared<Self>();
      *clone = s;
      return clone;
    }

    template<typename T>
    Ast operator()(T& node)
    {
      auto clone = std::make_shared<T>();
      *clone = node;
      *this << fields(*clone);
      return clone;
    }

    Clone& operator<<(Location& loc)
    {
      return *this;
    }

    template<typename T>
    Clone& operator<<(Node<T>& node)
    {
      node = std::static_pointer_cast<T>(dispatch(*this, node));
      return *this;
    }

    template<typename T>
    Clone& operator<<(Weak<T>& node)
    {
      node = std::static_pointer_cast<T>(dispatch(*this, node.lock()));
      return *this;
    }

    template<typename T>
    Clone& operator<<(List<T>& list)
    {
      for (auto& node : list)
        *this << node;

      return *this;
    }
  };

  Ast clone_ast(Substitutions& subs, Ast node, Ast self)
  {
    Clone clone(subs, self);
    return dispatch(clone, node);
  }

  Node<FunctionType> function_type(Lambda& lambda)
  {
    auto f = std::make_shared<FunctionType>();
    f->location = lambda.location;

    if (lambda.params.size() == 1)
    {
      f->left = lambda.params.front()->as<Param>().type;
    }
    else if (lambda.params.size() > 1)
    {
      auto t = std::make_shared<TupleType>();
      f->left = t;

      for (auto& p : lambda.params)
      {
        auto& pt = p->as<Param>().type;
        t->types.push_back(pt);
        t->location.extend(pt->location);
      }
    }

    if (f->left)
      f->location = f->left->location;

    f->right = lambda.result;
    return f;
  }

  Node<Type> receiver_type(Node<Type>& args)
  {
    if (!args)
      return {};

    if (args->kind() == Kind::TupleType)
      return args->as<TupleType>().types.front();

    return args;
  }

  Node<Type> receiver_self(Node<Type>& t, Node<Type>& self)
  {
    if (t->kind() != Kind::FunctionType)
      return {};

    auto f = t->as<FunctionType>();
    Node<Type> left;

    if (f.left->kind() == Kind::TupleType)
    {
      auto fl = f.left->as<TupleType>();
      auto fln = std::make_shared<TupleType>();
      fln->location = fl.location;
      fln->types = fl.types;
      fln->types.front() = dnf::conjunction(fl.types.front(), self);
      left = fln;
    }
    else
    {
      left = dnf::conjunction(f.left, self);
    }

    auto ft = std::make_shared<FunctionType>();
    ft->location = f.location;
    ft->left = left;
    ft->right = f.right;
    return ft;
  }

  Node<TypeRef> typeparamref(Node<TypeParam>& typeparam)
  {
    auto tn = std::make_shared<TypeName>();
    tn->location = typeparam->location;

    auto find = std::make_shared<LookupRef>();
    find->def = typeparam;

    auto tr = std::make_shared<TypeRef>();
    tr->location = typeparam->location;
    tr->typenames.push_back(tn);
    tr->lookup = find;

    return tr;
  }
}
