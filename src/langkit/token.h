// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

namespace langkit
{
  class Token;
  class NodeDef;
  using Node = std::shared_ptr<NodeDef>;
  using Binding = std::pair<Token, Node>;

  class TokenDef
  {
  public:
    using flag = uint32_t;

  private:
    const char* name_;
    flag fl;

  public:
    constexpr TokenDef(const char* name, flag fl = 0) : name_(name), fl(fl) {}

    TokenDef() = delete;
    TokenDef(const TokenDef&) = delete;
    TokenDef& operator=(const TokenDef&) = delete;

    Node operator()(Location loc) const;
    Binding operator=(const Token& token) const;
    Binding operator=(Node n) const;

    constexpr bool has(TokenDef::flag f) const
    {
      return (fl & f) != 0;
    }

    constexpr const char* name() const
    {
      return name_;
    }
  };

  class Token
  {
  private:
    const TokenDef* def;

  public:
    constexpr Token() : def(nullptr) {}
    constexpr Token(const Token& that) : def(that.def) {}
    constexpr Token(const TokenDef& def) : def(&def) {}

    operator Node() const;

    constexpr bool operator&(TokenDef::flag f) const
    {
      return (def->has(f)) != 0;
    }

    Node operator()(Location loc) const;

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

    constexpr const char* str() const
    {
      return def->name();
    }
  };

  inline Node TokenDef::operator()(Location loc) const
  {
    return Token(*this)(loc);
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
  }

  inline constexpr auto Invalid = TokenDef("invalid");
  inline constexpr auto Unclosed = TokenDef("unclosed");
  inline constexpr auto Group = TokenDef("group");
  inline constexpr auto File = TokenDef("file");
  inline constexpr auto Directory = TokenDef("directory");
}
