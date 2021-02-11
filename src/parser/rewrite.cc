// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "rewrite.h"

#include "dispatch.h"
#include "fields.h"

namespace verona::parser
{
  struct Rewrite
  {
    Ast prev;
    Ast next;
    bool ok;

    Rewrite() : ok(false) {}

    bool operator()(Ast& ast, Ast& with)
    {
      return false;
    }

    template<typename T>
    bool operator()(T& parent, Ast& prev, Ast& next)
    {
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
      if (!ok)
      {
        for (auto& node : list)
        {
          if (node == prev)
          {
            node = std::static_pointer_cast<T>(next);
            ok = true;
            break;
          }
        }
      }

      return *this;
    }
  };

  bool rewrite(AstPath& path, Ast node)
  {
    if (path.size() < 2)
      return false;

    auto& parent = path[path.size() - 2];
    auto& prev = path.back();

    Rewrite rewrite;
    dispatch(rewrite, parent, prev, node);

    if (!rewrite.ok)
      return false;

    path.pop_back();
    path.push_back(node);
    return true;
  }
}
