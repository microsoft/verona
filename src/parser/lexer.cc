#include "lexer.h"
#include "source.h"

namespace verona::parser
{
  const uint8_t X = 0; // Invalid
  const uint8_t W = 1; // Whitespace
  const uint8_t Y = 2; // Symbol
  const uint8_t S = 3; // String "
  const uint8_t Z = 4; // Builtin symbol .,()[]{};
  const uint8_t L = 5; // Slash /
  const uint8_t N = 6; // Number start 0123456789
  const uint8_t I = 7; // Ident start
  const uint8_t C = 8; // Colon :

  const uint8_t lookup[] =
  {
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
    X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,

    W, Y, S, Y, Y, Y, Y, Y, Z, Z, Y, Y, Z, Y, Z, L,
    N, N, N, N, N, N, N, N, N, N, C, Z, Y, Y, Y, Y,

    Y, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
    Y, I, I, I, I, I, I, I, I, I, I, Z, Y, Z, Y, I,

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

  bool is_digit(char c)
  {
    return
      ((c >= '0') && (c <= '9')) ||
      (c == '_');
  }

  bool is_hex(char c)
  {
    return
      ((c >= 'a') && (c <= 'f')) ||
      ((c >= 'A') && (c <= 'F')) ||
      is_digit(c);
  }

  bool is_binary(char c)
  {
    return
      ((c >= '0') && (c <= '1')) ||
      (c == '_');
  }

  Token consume_invalid(Source& source, size_t& i)
  {
    auto start = i;

    while (++i < source->contents.size())
    {
      if (lookup[source->contents[i]] != X)
        break;
    }

    return {TokenKind::Invalid, {source, start, i - 1}};
  }

  Token consume_symbol(Source& source, size_t& i)
  {
    auto start = i;

    while (++i < source->contents.size())
    {
      auto c = lookup[source->contents[i]];

      // Colons and slashes are valid symbol continuations.
      if ((c != Y) && (c != C) && (c != L))
        break;
    }

    return {TokenKind::Symbol, {source, start, i - 1}};
  }

  Token consume_builtin_symbol(Source& source, size_t& i)
  {
    auto kind = TokenKind::End;

    switch (source->contents[i])
    {
      case '.':
      {
        kind = TokenKind::Dot;
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
        kind = TokenKind::LBracket;
        break;
      }

      case '}':
      {
        kind = TokenKind::RBracket;
        break;
      }
    }

    return {kind, {source, i, i}};
  }

  void consume_line_comment(Source& source, size_t& i)
  {
    while (++i < source->contents.size())
    {
      if (source->contents[i] == '\n')
        break;
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

      switch (state)
      {
        case State::Slash:
        {
          state = State::Other;

          switch (c)
          {
            case '*':
            {
              depth++;
              break;
            }
          }
        }

        case State::Star:
        {
          state = State::Other;

          switch (c)
          {
            case '/':
            {
              if (--depth == 0)
                return;
              break;
            }
          }
        }

        case State::Other:
        {
          switch (c)
          {
            case '/':
            {
              state = State::Slash;
              break;
            }

            case '*':
            {
              state = State::Star;
              break;
            }
          }
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
        consume_line_comment(source, i);
        return true;
      }

      case '*':
      {
        consume_nested_comment(source, i);
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
      Float,
      Exponent1,
      Exponent2,
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
          state = State::Float;
        else if (!is_digit(c))
          break;
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
        if (is_digit(c) || (c == '-') || (c == '+'))
          state = State::Exponent2;
        else
          break;
      }
      else if (state == State::Exponent2)
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
      case State::Exponent1:
      case State::Exponent2:
        kind = TokenKind::Float;
        break;

      case State::Hex:
        kind = TokenKind::Hex;
        break;

      case State::Binary:
        kind = TokenKind::Binary;
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
        // Prime is the only symbol that's a valid ident continuation.
        if ((c != S) || (source->contents[i] != '\''))
          break;
      }
    }

    return {TokenKind::Ident, {source, start, i - 1}};
  }

  Token consume_colon(Source& source, size_t& i)
  {
    auto kind = TokenKind::Colon;
    auto start = i;

    while (++i < source->contents.size())
    {
      auto c = source->contents[i];

      if (lookup[c] == C)
      {
        kind = TokenKind::DoubleColon;
        ++i;
        break;
      }

      if (lookup[c] != S)
        break;

      kind = TokenKind::Symbol;
    }

    return {kind, {source, start, i - 1}};
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

        case S:
        {
          // TODO: string - how to do interpolated strings
          break;
        }

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
      }
    }

    return {TokenKind::End, {source, start, i - 1}};
  }
}
