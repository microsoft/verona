// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lit.h"

using namespace peg::udl;

namespace
{
  size_t nextline(const std::string& s, size_t pos = 0)
  {
    auto n = s.find("\n", pos);

    if (n != std::string::npos)
      n++;

    return n;
  }

  void remove_leading_blank_line(ast::Ast& ast)
  {
    if (ast->nodes.empty())
      return;

    auto node = ast->nodes.front();

    if (node->tag != "quote"_)
      return;

    auto& text = node->token;
    auto pos = text.find_first_not_of(" \f\r\t\v");

    if ((pos != std::string::npos) && (text[pos] == '\n'))
    {
      auto next = text.find_first_not_of(" \f\r\t\v", pos + 1);

      if (next == std::string::npos)
      {
        ast::remove(node);
      }
      else
      {
        auto trim = text.substr(pos + 1);
        auto quote = ast::token(node, "quote", trim);
        ast::replace(node, quote);
      }
    }
  }

  void remove_trailing_blank_line(ast::Ast& ast)
  {
    if (ast->nodes.empty())
      return;

    auto node = ast->nodes.back();

    if (node->tag != "quote"_)
      return;

    auto& text = node->token;
    auto pos = text.find_last_not_of(" \f\r\t\v");

    if ((pos != std::string::npos) && (text[pos] == '\n'))
    {
      if (pos == 0)
      {
        ast::remove(node);
      }
      else
      {
        auto trim = text.substr(0, pos);
        auto quote = ast::token(node, "quote", trim);
        ast::replace(node, quote);
      }
    }
  }

  size_t calc_indent(ast::Ast& ast)
  {
    if (ast->nodes.empty())
      return 0;

    // If the first node is not a quote, there is no indentation.
    if (ast->nodes.front()->tag != "quote"_)
      return 0;

    size_t indent = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < ast->nodes.size(); i++)
    {
      auto node = ast->nodes[i];

      // Only examine quote nodes. Unquote nodes may evaluate to whitespace, but
      // this is dynamic and not handled.
      if (node->tag != "quote"_)
        continue;

      // Start from the first character only on the first node. Otherwise start
      // from right after the first newline. This is because the previous node
      // is an unquote, and anything up to the first newline is trailing
      // whitespace, not leading whitespace.
      auto& text = node->token;
      size_t prev = (i > 0) ? nextline(text) : 0;

      while (prev != std::string::npos)
      {
        // Find the first non-whitespace character.
        auto pos = text.find_first_not_of(" \f\r\t\v", prev);
        size_t len = std::numeric_limits<size_t>::max();

        if (pos == std::string::npos)
        {
          // If this is not the last node, then the next node is an unquote.
          // That means this blank entry is leading whitespace. Otherwise, it is
          // an entirely blank line at the end of the string.
          if (i < (ast->nodes.size() - 1))
            len = text.size() - prev;
        }
        else if (text[pos] != '\n')
        {
          // Don't do this if we found a newline, as then it is a blank line
          // that is ignored for indent purposes.
          len = pos - prev;
        }

        if (len < indent)
          indent = len;

        // Find the beginning of the next line.
        prev = nextline(text, pos);
      }
    }

    // If all lines were blank, there is no indent.
    if (indent == std::numeric_limits<size_t>::max())
      return 0;

    return indent;
  }

  void trim_indent(ast::Ast& ast, size_t indent)
  {
    if (indent == 0)
      return;

    for (size_t i = 0; i < ast->nodes.size(); i++)
    {
      auto node = ast->nodes[i];

      if (node->tag != "quote"_)
        continue;

      // Start from the first character only on the first node. Otherwise
      // start from right after the first newline, since the previous node was
      // an unquote and anything up to the first newline is trailing text, not
      // leading text.
      auto& text = node->token;
      size_t prev = (i > 0) ? nextline(text) : 0;
      std::string s(text.substr(0, prev));

      while (prev != std::string::npos)
      {
        auto end = nextline(text, prev);
        auto start = prev + indent;

        // Handle blank lines shorter than the indent.
        if (end < start)
          start = end - 1;

        auto len = end - start;

        if (start < text.size())
          s.append(text.substr(start, len));

        prev = end;
      }

      auto quote = ast::token(node, "quote", s);
      ast::replace(node, quote);
    }
  }
}

namespace lit
{
  size_t hex(const std::string& src, size_t& i, size_t len)
  {
    size_t r = 0;

    for (size_t count = 0; count < len; count++)
    {
      if (i >= (src.size() - 1))
        return r;

      auto c = src[i + 1];

      if ((c >= '0') && (c <= '9'))
        c = c - '0';
      else if ((c >= 'A') && (c <= 'F'))
        c = c - 'A' + 10;
      else if ((c >= 'a') && (c <= 'f'))
        c = c - 'a' + 10;
      else
        return r;

      i++;
      r = (r << 4) + c;
    }

    return r;
  }

  std::string utf8(size_t v)
  {
    std::string s;

    if (v <= 0x7f)
    {
      s.push_back(v & 0x7f);
    }
    else if (v <= 0x7FF)
    {
      s.push_back(0xc0 | ((v >> 6) & 0xff));
      s.push_back(0x80 | (v & 0x3f));
    }
    else if (v <= 0xffff)
    {
      s.push_back(0xe0 | ((v >> 12) & 0xff));
      s.push_back(0x80 | ((v >> 6) & 0x3f));
      s.push_back(0x80 | (v & 0x3f));
    }
    else if (v < 0x10ffff)
    {
      s.push_back(0xf0 | ((v >> 18) & 0xff));
      s.push_back(0x80 | ((v >> 12) & 0x3f));
      s.push_back(0x80 | ((v >> 6) & 0x3f));
      s.push_back(0x80 | (v & 0x3f));
    }

    return s;
  }

  std::string crlf2lf(const std::string& src)
  {
    std::string s;
    size_t prev = 0;

    while (true)
    {
      auto pos = src.find("\r\n", prev);
      s.append(src.substr(prev, pos - prev));

      if (pos == std::string::npos)
        break;

      s.push_back('\n');
      prev = pos + 2;
    }

    return s;
  }

  void crlf2lf(ast::Ast& ast)
  {
    for (auto node : ast->nodes)
    {
      if (node->tag != "quote"_)
        continue;

      auto s = crlf2lf(node->token);
      auto lf = ast::token(node, "quote", s);
      ast::replace(node, lf);
    }
  }

  std::string escape(const std::string& src)
  {
    std::string dst;

    for (size_t i = 0; i < src.size(); i++)
    {
      auto c = src[i];

      if (c != '\\')
      {
        dst.push_back(c);
        continue;
      }

      auto n = src[++i];

      switch (n)
      {
        case 'a':
        {
          dst.push_back('\a');
          break;
        }

        case 'b':
        {
          dst.push_back('\b');
          break;
        }

        case 'e':
        {
          dst.push_back(0x1b);
          break;
        }

        case 'f':
        {
          dst.push_back('\f');
          break;
        }

        case 'n':
        {
          dst.push_back('\n');
          break;
        }

        case 'r':
        {
          dst.push_back('\r');
          break;
        }

        case 't':
        {
          dst.push_back('\t');
          break;
        }

        case 'v':
        {
          dst.push_back('\v');
          break;
        }

        case '\\':
        {
          dst.push_back('\\');
          break;
        }

        case '0':
        {
          dst.push_back('\0');
          break;
        }

        case 'x':
        {
          dst.append(utf8(hex(src, i, 2)));
          break;
        }

        case 'u':
        {
          dst.append(utf8(hex(src, i, 4)));
          break;
        }

        case 'U':
        {
          dst.append(utf8(hex(src, i, 6)));
          break;
        }

        default:
        {
          dst.push_back(n);
          break;
        }
      }
    }

    return dst;
  }

  void mangle_indent(ast::Ast& ast)
  {
    remove_leading_blank_line(ast);
    remove_trailing_blank_line(ast);
    trim_indent(ast, calc_indent(ast));
  }
}
