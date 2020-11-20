// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "source.h"

namespace verona::parser
{
  enum class TokenKind
  {
    Invalid,
    Dot,
    Comma,
    LParen,
    RParen,
    LSquare,
    RSquare,
    LBracket,
    RBracket,
    Semicolon,
    Colon,
    DoubleColon,
    Symbol,
    String,
    Int,
    Float,
    Hex,
    Binary,
    Ident,
    End
  };

  struct Token
  {
    TokenKind kind;
    Location location;
  };

  Token lex(Source& source, size_t& i, err::Errors& err);
}
