// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "rewrite.h"

#include "dispatch.h"
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

  bool rewrite(AstPath& path, size_t index, Ast next)
  {
    if (path.size() < 2)
      return false;

    auto& parent = path[path.size() - 2];
    auto& prev = path.back();

    Rewrite rewrite;
    dispatch(rewrite, parent, index, prev, next);

    if (!rewrite.ok)
      return false;

    path.pop_back();
    path.push_back(next);
    return true;
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

    Ast operator()(TypeRef& tr)
    {
      auto def = tr.def.lock();

      if (def->kind() == Kind::TypeParam)
      {
        auto tp = std::static_pointer_cast<TypeParam>(def);
        auto find = subs.find(tr.def);

        if (find != subs.end())
          return find->second;
      }

      auto clone = std::make_shared<TypeRef>();
      *clone = tr;

      for (auto& sub : clone->subs)
      {
        if (sub.second->kind() == Kind::TypeRef)
        {
          auto& tr = sub.second->as<TypeRef>();
          auto find = subs.find(tr.def);

          if (find != subs.end())
            sub.second = find->second;
        }
      }

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
      t->location = lambda.location;
      f->left = t;

      for (auto& p : lambda.params)
        t->types.push_back(p->as<Param>().type);
    }

    f->right = lambda.result;
    return f;
  }
}
