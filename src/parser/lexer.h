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
    String,
    Int,
    Float,
    Hex,
    Binary,
    True,
    False,

    // Keywords
    Module,
    Class,
    Interface,
    Type,
    Where,
    Static,
    Throws,
    If,
    Else,
    While,
    For,
    In,
    Match,
    When,
    Break,
    Continue,
    Return,
    Yield,
    Let,
    Var,
    New,

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
