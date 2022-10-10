// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

namespace langkit
{
  struct Token;
  using Binding = std::pair<Token, Node>;

  struct TokenDef
  {
    using flag = uint32_t;
    const char* name;
    flag fl;

    consteval TokenDef(const char* name, flag fl = 0) : name(name), fl(fl) {}

    TokenDef() = delete;
    TokenDef(const TokenDef&) = delete;

    operator Node() const;

    Binding operator=(const TokenDef& token) const;
    Binding operator=(const Token& token) const;
    Binding operator=(Node n) const;

    constexpr bool has(TokenDef::flag f) const
    {
      return (fl & f) != 0;
    }
  };

  struct Token
  {
    const TokenDef* def;

    constexpr Token() : def(nullptr) {}
    constexpr Token(const Token& that) : def(that.def) {}
    constexpr Token(const TokenDef& def) : def(&def) {}

    operator Node() const;

    constexpr bool operator&(TokenDef::flag f) const
    {
      return (def->has(f)) != 0;
    }

    Binding operator=(Node n) const
    {
      return {*this, n};
    }

    constexpr bool operator==(const Token& that) const
    {
      return def == that.def;
    }

    constexpr bool operator!=(const Token& that) const
    {
      return def != that.def;
    }

    constexpr bool operator<(const Token& that) const
    {
      return def < that.def;
    }

    constexpr bool operator>(const Token& that) const
    {
      return def > that.def;
    }

    constexpr bool operator<=(const Token& that) const
    {
      return def <= that.def;
    }

    constexpr bool operator>=(const Token& that) const
    {
      return def >= that.def;
    }

    constexpr bool in(std::initializer_list<Token> list) const
    {
      return std::find(list.begin(), list.end(), *this) != list.end();
    }

    constexpr const char* str() const
    {
      return def->name;
    }
  };

  inline Binding TokenDef::operator=(const TokenDef& token) const
  {
    return {Token(*this), Token(token)};
  }

  inline Binding TokenDef::operator=(const Token& token) const
  {
    return {Token(*this), token};
  }

  inline Binding TokenDef::operator=(Node n) const
  {
    return {Token(*this), n};
  }

  namespace flag
  {
    constexpr TokenDef::flag none = 0;
    constexpr TokenDef::flag print = 1 << 0;
    constexpr TokenDef::flag symtab = 1 << 1;
    constexpr TokenDef::flag defbeforeuse = 1 << 2;
    constexpr TokenDef::flag multidef = 1 << 3;
  }

  inline constexpr auto Invalid = TokenDef("invalid");
  inline constexpr auto Unclosed = TokenDef("unclosed");
  inline constexpr auto Top = TokenDef("top", flag::symtab);
  inline constexpr auto Group = TokenDef("group");
  inline constexpr auto File = TokenDef("file");
  inline constexpr auto Directory = TokenDef("directory");
  inline constexpr auto Seq = TokenDef("seq");
  inline constexpr auto Lift = TokenDef("lift");
  inline constexpr auto Error = TokenDef("error");
  inline constexpr auto ErrorMsg = TokenDef("errormsg", flag::print);
  inline constexpr auto ErrorAst = TokenDef("errorast");
}
