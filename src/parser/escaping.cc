// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "escaping.h"

#include <sstream>

namespace
{
  bool is_binary(char c)
  {
    return (c == '0') || (c == '1');
  }

  bool is_hex(char c)
  {
    return ((c >= '0') && (c <= '9')) || ((c >= 'a') && (c <= 'f')) ||
      ((c >= 'A') && (c <= 'F'));
  }

  bool is_xescape(const std::string_view& s, size_t i)
  {
    return ((i + 1) < s.size()) && is_hex(s[i]) && is_hex(s[i + 1]);
  }

  bool is_uescape(const std::string_view& s, size_t i)
  {
    return ((i + 3) < s.size()) && is_hex(s[i]) && is_hex(s[i + 1]) &&
      is_hex(s[i + 2]) && is_hex(s[i + 3]);
  }

  bool is_Uescape(const std::string_view& s, size_t i)
  {
    if ((i + 5) >= s.size())
      return false;

    if ((s[i] == '1') && (s[i + 1] != '0'))
      return false;

    return is_binary(s[i]) && is_hex(s[i + 1]) && is_hex(s[i + 2]) &&
      is_hex(s[i + 3]) && is_hex(s[i + 4]) && is_hex(s[i + 5]);
  }

  bool is_escape(const std::string_view& s, size_t& i)
  {
    // Valid escapes are:
    // \' \" \\ \a \b \e \f \n \r \t \v \0 \xHH \uHHHH \UHHHHHH
    if (s[i] != '\\')
      return false;

    if ((i + 1) >= s.size())
      return false;

    switch (s[i + 1])
    {
      case '\'':
      case '\"':
      case '\\':
      case 'a':
      case 'b':
      case 'e':
      case 'f':
      case 'n':
      case 'r':
      case 't':
      case 'v':
      case '0':
      {
        i++;
        return true;
      }

      case 'x':
      {
        bool ok = is_xescape(s, i + 2);

        if (ok)
          i += 3;

        return ok;
      }

      case 'u':
      {
        bool ok = is_uescape(s, i + 2);

        if (ok)
          i += 5;

        return ok;
      }

      case 'U':
      {
        bool ok = is_Uescape(s, i + 2);

        if (ok)
          i += 7;

        return ok;
      }

      default:
        return false;
    }
  }

  bool is_utf8cont(char c)
  {
    return (c & 0xC0) == 0x80;
  }

  bool is_utf8(const std::string_view& s, size_t& i)
  {
    if ((s[i] & 0x80) == 0x00)
      return true;

    if ((s[i] & 0xE0) == 0xC0)
    {
      bool ok = ((i + 1) < s.size()) && is_utf8cont(s[i + 1]);

      if (ok)
        i += 1;

      return ok;
    }

    if ((s[i] & 0xF0) == 0xE0)
    {
      bool ok =
        ((i + 2) < s.size()) && is_utf8cont(s[i + 1]) && is_utf8cont(s[i + 2]);

      if (ok)
        i += 2;

      return ok;
    }

    if ((s[i] & 0xF8) == 0xF0)
    {
      bool ok = ((i + 3) < s.size()) && is_utf8cont(s[i + 1]) &&
        is_utf8cont(s[i + 2]) && is_utf8cont(s[i + 3]);

      if (ok)
        i += 3;

      return ok;
    }

    return false;
  }

  char to_hex(char c)
  {
    return (c < 10) ? c + '0' : c + 'A' - 10;
  }

  char from_hex(char c)
  {
    if ((c >= '0') && (c <= '9'))
      return c - '0';

    if ((c >= 'a') && (c <= 'f'))
      return c - 'a' + 10;

    if ((c >= 'A') && (c <= 'F'))
      return c - 'A' + 10;

    return 0;
  }

  struct utf8
  {
    uint32_t h;

    utf8(const std::string_view& s, size_t& i) : h(0)
    {
      uint8_t len = 0;

      switch (s[i])
      {
        case 'x':
        {
          if (is_xescape(s, i + 1))
            len = 2;
          break;
        }

        case 'u':
        {
          if (is_uescape(s, i + 1))
            len = 4;
          break;
        }

        case 'U':
        {
          if (is_Uescape(s, i + 1))
            len = 6;
          break;
        }

        default:
          return;
      }

      uint8_t shift = len * 4;

      for (; len > 0; --len)
      {
        shift -= 4;
        h |= from_hex(s[++i]) << shift;
      }
    }
  };

  std::ostream& operator<<(std::ostream& out, const utf8& h)
  {
    if (h.h < 0x80)
      return out << uint8_t(h.h);

    if (h.h < 0x800)
      return out << uint8_t(0xC0 | (h.h >> 6)) << uint8_t(0x80 | (h.h & 0x3F));

    if (h.h < 0x10000)
      return out << uint8_t(0xE0 | (h.h >> 12))
                 << uint8_t(0x80 | ((h.h >> 6) & 0x3F))
                 << uint8_t(0x80 | (h.h & 0x3F));

    if (h.h < 0x110000)
      return out << uint8_t(0xF0 | (h.h >> 18))
                 << uint8_t(0x80 | ((h.h >> 12) & 0x3F))
                 << uint8_t(0x80 | ((h.h >> 6) & 0x3F))
                 << uint8_t(0x80 | (h.h & 0x3F));

    return out;
  }

  std::string crlf2lf(const std::string_view& src)
  {
    std::ostringstream ss;
    std::string_view s = src;

    while (true)
    {
      auto pos = s.find("\r\n");
      ss << s.substr(0, pos);

      if (pos == std::string::npos)
        break;

      ss << '\n';
      s = s.substr(pos + 2);
    }

    return ss.str();
  }

  std::string unescape(const std::string_view& s)
  {
    std::ostringstream ss;
    auto backslash = false;

    for (size_t i = 0; i < s.size(); i++)
    {
      auto c = s[i];

      if (!backslash)
      {
        if (c == '\\')
          backslash = true;
        else
          ss << c;
      }
      else
      {
        backslash = false;

        switch (c)
        {
          case '\'':
          case '\"':
          case '\\':
            ss << c;
            break;

          case 'a':
            ss << '\a';
            break;

          case 'b':
            ss << '\b';
            break;

          case 'e':
            ss << char(0x1B);
            break;

          case 'f':
            ss << '\f';
            break;

          case 'n':
            ss << '\n';
            break;

          case 'r':
            ss << '\r';
            break;

          case 't':
            ss << '\t';
            break;

          case 'v':
            ss << '\v';
            break;

          case '0':
            ss << '\0';
            break;

          case 'x':
          case 'u':
          case 'U':
          {
            ss << utf8(s, i);
            break;
          }

          default:
          {
            ss << c;
            break;
          }
        }
      }
    }

    return ss.str();
  }

  std::string_view trimblanklines(const std::string_view& src)
  {
    // Remove leading blank line.
    std::string_view s = src;
    auto pos = s.find_first_not_of(" \f\r\t\v");

    if ((pos != std::string::npos) && (s[pos] == '\n'))
      s = s.substr(pos + 1);

    // Remove trailing blank line.
    pos = s.find_last_not_of(" \f\r\t\v");

    if ((pos != std::string::npos) && (s[pos] == '\n'))
      s = s.substr(0, pos);

    return s;
  }
}

namespace verona::parser
{
  bool is_escaped(const std::string_view& s)
  {
    for (size_t i = 0; i < s.size(); i++)
    {
      if (!is_escape(s, i) && !is_utf8(s, i))
        return false;
    }

    return true;
  }

  bool is_unescaped(const std::string_view& s)
  {
    for (size_t i = 0; i < s.size(); i++)
    {
      if (!is_utf8(s, i))
        return false;
    }

    return true;
  }

  std::string escapedstring(const std::string_view& s)
  {
    return unescape(trimblanklines(crlf2lf(s)));
  }

  std::string unescapedstring(const std::string_view& s)
  {
    return std::string(trimblanklines(crlf2lf(s)));
  }

  std::string escape(const std::string_view& s)
  {
    // Mostly JSON-style escaping.
    std::ostringstream ss;

    for (size_t i = 0; i < s.size(); i++)
    {
      uint8_t c = s[i];

      switch (c)
      {
        case '\'':
          ss << "\\\'";
          break;

        case '\"':
          ss << "\\\"";
          break;

        case '\\':
        {
          ss << "\\\\";
          break;
        }

        case '\a':
          ss << "\\a";
          break;

        case '\b':
          ss << "\\b";
          break;

        case 0x1B:
          ss << "\\e";
          break;

        case '\f':
          ss << "\\f";
          break;

        case '\n':
          ss << "\\n";
          break;

        case '\r':
          ss << "\\r";
          break;

        case '\t':
          ss << "\\t";
          break;

        case '\v':
          ss << "\\v";
          break;

        case '\0':
          ss << "\\0";
          break;

        default:
        {
          if ((c >= ' ') && (c <= '~'))
          {
            ss << c;
          }
          else if (c <= 0x7F)
          {
            ss << "\\x" << to_hex(c >> 4) << to_hex(c & 0x0F);
          }
          else if ((c & 0xE0) == 0xC0)
          {
            ss << s.substr(i, 2);
            i++;
          }
          else if ((c & 0xF0) == 0xE0)
          {
            ss << s.substr(i, 3);
            i += 2;
          }
          else if ((c & 0xF8) == 0xF0)
          {
            ss << s.substr(i, 4);
            i += 3;
          }
          break;
        }
      }
    }

    return ss.str();
  }
}
