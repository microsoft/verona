// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lexer.h"

#include "escaping.h"

namespace verona::parser
{
  constexpr uint8_t X = 0; // Invalid
  constexpr uint8_t W = 1; // Whitespace sp\t\r\n
  constexpr uint8_t Y = 2; // Symbol
  constexpr uint8_t Q = 3; // Quote '"
  constexpr uint8_t Z = 4; // Builtin symbol .,()[]{};
  constexpr uint8_t L = 5; // Slash /
  constexpr uint8_t N = 6; // Number start 0123456789
  constexpr uint8_t I = 7; // Ident start
  constexpr uint8_t C = 8; // Colon :
  constexpr uint8_t E = 9; // Equal =
  constexpr uint8_t Eof = 255; // End of file

  constexpr uint8_t lookup[] = {
    X, X, X, X, X, X, X, X, X, W, W, X, X, W, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,

    W, Y, Q, Y, Y, Y, Y, Q, Z, Z, Y, Y, Z, Y, Z, L,
    N, N, N, N, N, N, N, N, N, N, C, Z, Y, E, Y, Y,

    Y, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
    I, I, I, I, I, I, I, I, I, I, I, Z, Y, Z, Y, I,

    Y, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
    I, I, I, I, I, I, I, I, I, I, I, Z, Y, Z, Y, X,

    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,

    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,

    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,

    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
  };

  struct Keyword
  {
    const char* text;
    TokenKind kind;
  };

  constexpr Keyword keywords[] = {{"module", TokenKind::Module},
                                  {"class", TokenKind::Class},
                                  {"interface", TokenKind::Interface},
                                  {"type", TokenKind::Type},
                                  {"using", TokenKind::Using},
                                  {"try", TokenKind::Try},
                                  {"catch", TokenKind::Catch},
                                  {"throw", TokenKind::Throw},
                                  {"match", TokenKind::Match},
                                  {"when", TokenKind::When},
                                  {"let", TokenKind::Let},
                                  {"var", TokenKind::Var},
                                  {"new", TokenKind::New},
                                  {"iso", TokenKind::Iso},
                                  {"mut", TokenKind::Mut},
                                  {"imm", TokenKind::Imm},
                                  {"Self", TokenKind::Self},
                                  {"true", TokenKind::Bool},
                                  {"false", TokenKind::Bool},
                                  {nullptr, TokenKind::Invalid}};

  bool is_digit(char c)
  {
    return ((c >= '0') && (c <= '9')) || (c == '_');
  }

  bool is_hex(char c)
  {
    return ((c >= 'a') && (c <= 'f')) || ((c >= 'A') && (c <= 'F')) ||
      is_digit(c);
  }

  bool is_binary(char c)
  {
    return ((c >= '0') && (c <= '1')) || (c == '_');
  }

  uint8_t next(Source& source, size_t& i)
  {
    if ((i + 1) < source->contents.size())
      return lookup[source->contents[++i]];

    return Eof;
  }

  Token consume_invalid(Source& source, size_t& i)
  {
    auto start = i;
    while (next(source, i) == X)
    {
    }
    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  Token consume_symbol(Source& source, size_t& i)
  {
    auto start = i;

    while (true)
    {
      // Colons, slashes, and equals are valid symbol continuations.
      switch (next(source, i))
      {
        case Y:
        case L:
        case C:
        case E:
          break;

        default:
          return {TokenKind::Symbol, {source, start, i - 1}};
      }
    }
  }

  Token consume_builtin_symbol(Source& source, size_t& i)
  {
    TokenKind kind;
    auto start = i;

    switch (source->contents[i])
    {
      case '.':
      {
        if (
          ((i + 2) < source->contents.size()) &&
          (source->contents[i + 1] == '.') && (source->contents[i + 2] == '.'))
        {
          kind = TokenKind::Ellipsis;
          i += 2;
        }
        else
        {
          kind = TokenKind::Dot;
        }
        break;
      }

      case ',':
      {
        kind = TokenKind::Comma;
        break;
      }

      case '(':
      {
        kind = TokenKind::LParen;
        break;
      }

      case ')':
      {
        kind = TokenKind::RParen;
        break;
      }

      case '[':
      {
        kind = TokenKind::LSquare;
        break;
      }

      case ']':
      {
        kind = TokenKind::RSquare;
        break;
      }

      case '{':
      {
        kind = TokenKind::LBrace;
        break;
      }

      case '}':
      {
        kind = TokenKind::RBrace;
        break;
      }

      case ';':
      {
        kind = TokenKind::Semicolon;
        break;
      }

      default:
        abort();
    }

    return {kind, {source, start, i++}};
  }

  Token consume_character_literal(Source& source, size_t& i)
  {
    auto start = i;
    bool backslash = false;

    while (++i < source->contents.size())
    {
      switch (source->contents[i])
      {
        case '\\':
        {
          backslash = true;
          break;
        }

        case '\'':
        {
          if (!backslash)
          {
            Location loc{source, start + 1, i++ - 1};

            if (!is_escaped(loc.view()))
              return {TokenKind::Invalid, {source, start, i - 1}};

            return {TokenKind::Character, loc};
          }

          backslash = false;
          break;
        }

        default:
        {
          backslash = false;
          break;
        }
      }
    }

    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  Token consume_escaped_string(Source& source, size_t& i)
  {
    auto start = i;
    bool backslash = false;

    while (++i < source->contents.size())
    {
      switch (source->contents[i])
      {
        case '\\':
        {
          backslash = true;
          break;
        }

        case '\"':
        {
          if (!backslash)
          {
            Location loc{source, start + 1, i++ - 1};

            if (!is_escaped(loc.view()))
              return {TokenKind::Invalid, {source, start, i - 1}};

            return {TokenKind::EscapedString, loc};
          }

          backslash = false;
          break;
        }

        default:
        {
          backslash = false;
          break;
        }
      }
    }

    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  Token consume_unescaped_string(Source& source, size_t& i, size_t len)
  {
    enum class State
    {
      Nesting,
      Terminating,
    };

    auto start = i - len;
    auto state = State::Nesting;
    size_t count = 0;
    size_t depth = 1;

    while (++i < source->contents.size())
    {
      switch (source->contents[i])
      {
        case '\"':
        {
          if (state == State::Nesting)
          {
            if (count == len)
              depth++;
            else
              state = State::Terminating;
          }
          else
          {
            state = State::Nesting;
          }

          count = 0;
          break;
        }

        case '\'':
        {
          count++;

          if ((count == len) && (state == State::Terminating))
          {
            depth--;
            count = 0;

            if (depth == 0)
            {
              Location loc{source, start + len + 1, i++ - len - 1};

              if (!is_unescaped(loc.view()))
                return {TokenKind::Invalid, {source, start, i - 1}};

              return {TokenKind::UnescapedString, loc};
            }
          }
          break;
        }

        default:
        {
          state = State::Nesting;
          count = 0;
          break;
        }
      }
    }

    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  Token consume_string(Source& source, size_t& i)
  {
    // '* " is an unescaped string
    // " is an escaped string
    // ' is a character literal
    if (source->contents[i] == '\"')
      return consume_escaped_string(source, i);

    auto start = i;

    while (++i < source->contents.size())
    {
      switch (source->contents[i])
      {
        case '\'':
          continue;

        case '\"':
          return consume_unescaped_string(source, i, i - start);
      }
      break;
    }

    if ((i - start) == 1)
      return consume_character_literal(source, --i);

    // It's an empty character literal.
    i = start + 2;
    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  void consume_line_comment(Source& source, size_t& i)
  {
    while (++i < source->contents.size())
    {
      if (source->contents[i] == '\n')
      {
        i++;
        break;
      }
    }
  }

  void consume_nested_comment(Source& source, size_t& i)
  {
    enum class State
    {
      Slash,
      Star,
      Other,
    };

    auto state = State::Other;
    size_t depth = 1;

    while (++i < source->contents.size())
    {
      auto c = source->contents[i];

      switch (c)
      {
        case '/':
        {
          if (state == State::Star)
          {
            state = State::Other;

            if (--depth == 0)
            {
              i++;
              return;
            }
          }
          else
          {
            state = State::Slash;
          }
          break;
        }

        case '*':
        {
          if (state == State::Slash)
          {
            state = State::Other;
            depth++;
          }
          else
          {
            state = State::Star;
          }
          break;
        }

        default:
        {
          state = State::Other;
          break;
        }
      }
    }
  }

  bool consume_comment(Source& source, size_t& i)
  {
    if ((i + 1) >= source->contents.size())
      return false;

    auto start = i;
    auto c = source->contents[++i];

    switch (c)
    {
      case '/':
      {
        consume_line_comment(source, ++i);
        return true;
      }

      case '*':
      {
        consume_nested_comment(source, ++i);
        return true;
      }
    }

    i = start;
    return false;
  }

  Token consume_number(Source& source, size_t& i)
  {
    enum class State
    {
      LeadingZero,
      Int,
      HasDot,
      Float,
      Exponent1,
      Exponent2,
      Exponent3,
      Hex,
      Binary,
    };

    auto start = i;
    auto state = State::Int;

    if (source->contents[i] == '0')
      state = State::LeadingZero;

    while (++i < source->contents.size())
    {
      auto c = source->contents[i];

      if (state == State::LeadingZero)
      {
        if (c == 'x')
          state = State::Hex;
        else if (c == 'b')
          state = State::Binary;
        else if (c == '.')
          state = State::Float;
        else if (is_digit(c))
          state = State::Int;
        else
          break;
      }
      else if (state == State::Int)
      {
        if (c == '.')
          state = State::HasDot;
        else if (!is_digit(c))
          break;
      }
      else if (state == State::HasDot)
      {
        if (is_digit(c))
        {
          state = State::Float;
        }
        else
        {
          // Don't consume the dot.
          --i;
          state = State::Int;
          break;
        }
      }
      else if (state == State::Float)
      {
        if (c == 'e')
          state = State::Exponent1;
        else if (!is_digit(c))
          break;
      }
      else if (state == State::Exponent1)
      {
        if ((c == '-') || (c == '+'))
          state = State::Exponent2;
        else if (is_digit(c))
          state = State::Exponent3;
        else
        {
          // Don't consume the e.
          --i;
          state = State::Float;
          break;
        }
      }
      else if (state == State::Exponent2)
      {
        if (is_digit(c))
        {
          state = State::Exponent3;
        }
        else
        {
          // Don't consume the e or the +/-.
          i -= 2;
          state = State::Float;
          break;
        }
      }
      else if (state == State::Exponent3)
      {
        if (!is_digit(c))
          break;
      }
      else if (state == State::Hex)
      {
        if (!is_hex(c))
          break;
      }
      else if (state == State::Binary)
      {
        if (!is_binary(c))
          break;
      }
    }

    auto kind = TokenKind::Invalid;

    switch (state)
    {
      case State::LeadingZero:
      case State::Int:
        kind = TokenKind::Int;
        break;

      case State::Float:
      case State::Exponent3:
        kind = TokenKind::Float;
        break;

      case State::Hex:
        kind = TokenKind::Hex;
        break;

      case State::Binary:
        kind = TokenKind::Binary;
        break;

      default:
        // Shouldn't reach this, leave it as an invalid token.
        break;
    }

    return {kind, {source, start, i - 1}};
  }

  Token consume_ident(Source& source, size_t& i)
  {
    auto start = i;

    while (++i < source->contents.size())
    {
      auto c = lookup[source->contents[i]];

      // Idents and numbers are valid ident continuations.
      if ((c != I) && (c != N))
      {
        // Prime is the only other thing that's a valid ident continuation.
        if ((c != Q) || (source->contents[i] != '\''))
          break;
      }
    }

    Token tok{TokenKind::Ident, {source, start, i - 1}};

    for (auto kw = &keywords[0]; kw->text; kw++)
    {
      if (tok.location == kw->text)
      {
        tok.kind = kw->kind;
        break;
      }
    }

    return tok;
  }

  Token consume_colon(Source& source, size_t& i)
  {
    auto kind = TokenKind::Colon;
    auto start = i;

    if (++i < source->contents.size())
    {
      switch (lookup[source->contents[i]])
      {
        case C:
          return {TokenKind::DoubleColon, {source, start, ++i - 1}};

        case Y:
        case E:
          return consume_symbol(source, --i);

        default:
          break;
      }
    }

    return {TokenKind::Colon, {source, start, i - 1}};
  }

  Token consume_equal(Source& source, size_t& i)
  {
    auto kind = TokenKind::Equals;
    auto start = i;

    if (++i < source->contents.size())
    {
      if (source->contents[i] == '>')
        return {TokenKind::FatArrow, {source, start, ++i - 1}};

      switch (lookup[source->contents[i]])
      {
        case Y:
        case C:
        case E:
          return consume_symbol(source, --i);

        default:
          break;
      }
    }

    return {TokenKind::Equals, {source, start, i - 1}};
  }

  Token lex(Source& source, size_t& i)
  {
    auto start = i;

    while (i < source->contents.size())
    {
      switch (lookup[source->contents[i]])
      {
        case X:
          return consume_invalid(source, i);

        case W:
        {
          // Skip whitespace.
          i++;
          break;
        }

        case Y:
          return consume_symbol(source, i);

        case Q:
          return consume_string(source, i);

        case Z:
          return consume_builtin_symbol(source, i);

        case L:
        {
          if (consume_comment(source, i))
            continue;

          return consume_symbol(source, i);
        }

        case N:
          return consume_number(source, i);

        case I:
          return consume_ident(source, i);

        case C:
          return consume_colon(source, i);

        case E:
          return consume_equal(source, i);
      }
    }

    return {TokenKind::End, {source, start, i - 1}};
  }
}
