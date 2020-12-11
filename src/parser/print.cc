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

  PrettyStream& operator<<(PrettyStream& out, Token& token)
  {
    switch (token.kind)
    {
      case TokenKind::EscapedString:
        return out << start("string") << sep << q
                   << escape(escapedstring(token.location.view())) << q << end;

      case TokenKind::UnescapedString:
        return out << start("string") << sep << q
                   << escape(unescapedstring(token.location.view())) << q
                   << end;

      case TokenKind::Character:
        return out << start("char") << sep << q
                   << escape(escapedstring(token.location.view())) << q << end;

      case TokenKind::Int:
        return out << start("int") << token.location << end;

      case TokenKind::Float:
        return out << start("float") << token.location << end;

      case TokenKind::Hex:
        return out << start("hex") << token.location << end;

      case TokenKind::Binary:
        return out << start("binary") << token.location << end;

      default:
        break;
    }

    return out << token.location;
  }

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, T& node)
  {
    return out << start(kindname(node.kind())) << fields(node) << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Constant& node)
  {
    return out << fields(node);
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
