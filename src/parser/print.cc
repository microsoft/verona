// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "print.h"

#include "dispatch.h"
#include "escaping.h"
#include "fields.h"
#include "pretty.h"

namespace verona::parser
{
  // Forward reference to break cycles.
  PrettyStream& operator<<(PrettyStream& out, const Ast& node);

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, Node<T>& node)
  {
    return out << static_cast<Ast>(node);
  }

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, std::vector<T>& vec)
  {
    if (vec.size() > 0)
    {
      out << sep << start("", '[');
      out << vec[0];

      for (size_t i = 1; i < vec.size(); i++)
        out << sep << vec[i];

      out << sep << endtoken(']');
    }
    else
    {
      out << sep << "[]";
    }

    return out;
  }

  PrettyStream& operator<<(PrettyStream& out, Location& loc)
  {
    if (!loc.source)
      return out << sep << "()";

    return out << sep << loc.view();
  }

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, T& node)
  {
    return out << start(kindname(node.kind())) << fields(node) << end;
  }

  PrettyStream& operator<<(PrettyStream& out, EscapedString& node)
  {
    return out << start(kindname(node.kind())) << sep << q
               << escape(escapedstring(node.location.view())) << q << end;
  }

  PrettyStream& operator<<(PrettyStream& out, UnescapedString& node)
  {
    return out << start(kindname(node.kind())) << sep << q
               << escape(unescapedstring(node.location.view())) << q << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Character& node)
  {
    return out << start(kindname(node.kind())) << sep << q
               << escape(escapedstring(node.location.view())) << q << end;
  }

  struct Print
  {
    PrettyStream& operator()(PrettyStream& out)
    {
      return out << sep << "()";
    }

    template<typename T>
    PrettyStream& operator()(T& node, PrettyStream& out)
    {
      return out << node;
    }
  };

  PrettyStream& operator<<(PrettyStream& out, const Ast& node)
  {
    Print print;
    return dispatch(print, node, out);
  }

  std::ostream& operator<<(std::ostream& out, const pretty& pret)
  {
    PrettyStream ss(out, pret.width);
    ss << pret.node;
    ss.flush();
    return out;
  }
}
