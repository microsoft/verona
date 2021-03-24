// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

namespace verona::parser
{
  enum class TokenKind
  {
    Invalid,

    // Builtin symbols
    Dot,
    Ellipsis,
    Comma,
    LParen,
    RParen,
    LSquare,
    RSquare,
    LBrace,
    RBrace,
    Semicolon,
    Colon,
    DoubleColon,
    FatArrow,
    Equals,

    // Constants
    EscapedString,
    UnescapedString,
    Character,
    Int,
    Float,
    Hex,
    Binary,
    Bool,

    // Keywords
    Module,
    Class,
    Interface,
    Type,
    Using,
    Try,
    Catch,
    Throw,
    Match,
    When,
    Let,
    Var,
    New,

    // Types
    Iso,
    Mut,
    Imm,
    Self,

    // Symbols and identifiers
    Symbol,
    Ident,

    End
  };

  struct Token
  {
    TokenKind kind;
    Location location;
  };

  Token lex(Source& source, size_t& i);
}
