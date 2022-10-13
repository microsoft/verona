// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace sample
{
  constexpr size_t restart = 0;
  const std::initializer_list<Token> terminators = {Equals, List};

  Parse parser()
  {
    Parse p(depth::subdirectories);
    auto depth = std::make_shared<size_t>(0);
    auto indent = std::make_shared<std::vector<size_t>>();
    indent->push_back(restart);

    p.prefile(
      [](auto& p, auto& path) { return path::extension(path) == "verona"; });

    p.predir([](auto& p, auto& path) {
      static auto re = std::regex(
        ".*/[_[:alpha:]][_[:alnum:]]*/$", std::regex_constants::optimize);
      return std::regex_match(path, re);
    });

    p.postparse([](auto& p, const std::string& path, auto ast) {
      auto stdlib = path::directory(path::executable()) + "std/";
      if (path != stdlib)
        ast->push_back(p.sub_parse(stdlib));
    });

    p.postfile([indent, depth](auto& p, const std::string& path, auto ast) {
      *depth = 0;
      indent->clear();
      indent->push_back(restart);
    });

    p("start",
      {
        // Blank lines terminate.
        "\n(?:[[:blank:]]*\n)+([[:blank:]]*)" >>
          [indent](auto& m) {
            indent->back() = m.match().length(1);
            m.term(terminators);
          },

        // A newline that starts a brace block doesn't terminate.
        "\n([[:blank:]]*(\\{)[[:blank:]]*)" >>
          [indent](auto& m) {
            indent->push_back(m.match().length(1));
            m.push(Brace, 2);
          },

        // A newline sometimes terminates.
        "\n([[:blank:]]*)" >>
          [indent](auto& m) {
            auto col = m.match().length(1);
            auto prev = indent->back();

            // If following a brace, don't terminate, but reset indentation.
            if (m.previous(Brace))
            {
              indent->back() = col;
              return;
            }

            // Don't terminate and don't reset indentation if:
            // * in an equals or list
            // * in a group and indented
            if (m.in(Equals) || m.in(List) || (m.in(Group) && (col > prev)))
              return;

            // Otherwise, terminate and reset indentation.
            m.term(terminators);
            indent->back() = col;
          },

        // Whitespace between tokens.
        "[[:blank:]]+" >> [](auto& m) {},

        // Terminator.
        ";" >> [](auto& m) { m.term(terminators); },

        // FatArrow.
        "=>" >>
          [indent](auto& m) {
            indent->back() = m.linecol().second + 1;
            m.term(terminators);
            m.add(FatArrow);
            m.term(terminators);
          },

        // Equals.
        "=" >> [](auto& m) { m.seq(Equals); },

        // List.
        "," >> [](auto& m) { m.seq(List, {Equals}); },

        // Parens.
        "(\\()[[:blank:]]*" >>
          [indent](auto& m) {
            indent->push_back(m.linecol().second + m.match().length());
            m.push(Paren, 1);
          },

        "\\)" >>
          [indent](auto& m) {
            indent->pop_back();
            m.term(terminators);
            m.pop(Paren);
          },

        // Square brackets.
        "(\\[)[[:blank:]]*" >>
          [indent](auto& m) {
            indent->push_back(m.linecol().second + m.match().length());
            m.push(Square, 1);
          },

        "\\]" >>
          [indent](auto& m) {
            indent->pop_back();
            m.term(terminators);
            m.pop(Square);
          },

        // Curly braces.
        "(\\{)[[:blank:]]*" >>
          [indent](auto& m) {
            indent->push_back(m.linecol().second + m.match().length());
            m.push(Brace, 1);
          },

        "\\}" >>
          [indent](auto& m) {
            indent->pop_back();
            m.term(terminators);
            m.pop(Brace);
          },

        // Bool.
        "(?:true|false)\\b" >> [](auto& m) { m.add(Bool); },

        // Hex float.
        "0x[[:xdigit:]]+\\.[[:xdigit:]]+(?:p[+-][[:digit:]]+)?\\b" >>
          [](auto& m) { m.add(HexFloat); },

        // Hex.
        "0x[_[:xdigit:]]+\\b" >> [](auto& m) { m.add(Hex); },

        // Bin.
        "0b[_01]+\\b" >> [](auto& m) { m.add(Bin); },

        // Float.
        "[[]:digit:]]+\\.[[:digit:]]+(?:e[+-]?[[:digit:]]+)?\\b" >>
          [](auto& m) { m.add(Float); },

        // Int.
        "[[:digit:]]+\\b" >> [](auto& m) { m.add(Int); },

        // Escaped string.
        "\"(?:\\\"|[^\"])*\"" >> [](auto& m) { m.add(Escaped); },

        // Unescaped string.
        "('+)\"[\\s\\S]*?\"\\1" >> [](auto& m) { m.add(String); },

        // Character literal.
        "'[^']*'" >> [](auto& m) { m.add(Char); },

        // Line comment.
        "//[^\n]*" >> [](auto& m) {},

        // Nested comment.
        "/\\*" >>
          [depth](auto& m) {
            ++(*depth);
            m.mode("comment");
          },

        // Keywords.
        "package\\b" >> [](auto& m) { m.add(Package); },
        "use\\b" >> [](auto& m) { m.add(Use); },
        "type\\b" >> [](auto& m) { m.add(TypeAlias); },
        "class\\b" >> [](auto& m) { m.add(Class); },
        "var\\b" >> [](auto& m) { m.add(Var); },
        "let\\b" >> [](auto& m) { m.add(Let); },
        "ref\\b" >> [](auto& m) { m.add(Ref); },
        "throw\\b" >> [](auto& m) { m.add(Throw); },
        "iso\\b" >> [](auto& m) { m.add(Iso); },
        "imm\\b" >> [](auto& m) { m.add(Imm); },
        "mut\\b" >> [](auto& m) { m.add(Mut); },

        // Don't care.
        "_(?![_[:alnum:]])" >> [](auto& m) { m.add(DontCare); },

        // Reserve a sequence of underscores.
        "_(?:_)+(?![[:alnum:]])" >> [](auto& m) { m.add(Invalid); },

        // Identifier.
        "[_[:alpha:]][_[:alnum:]]*\\b" >> [](auto& m) { m.add(Ident); },

        // Ellipsis.
        "\\.\\.\\." >> [](auto& m) { m.add(Ellipsis); },

        // Dot.
        "\\." >> [](auto& m) { m.add(Dot); },

        // Double colon.
        "::" >> [](auto& m) { m.add(DoubleColon); },

        // Colon.
        ":" >> [](auto& m) { m.add(Colon); },

        // Symbol. Reserved: "'(),.:;[]_{}
        "[!#$%&*+-/<=>?@\\^`|~]+" >> [](auto& m) { m.add(Symbol); },
      });

    p("comment",
      {
        "(?:[^\\*]|\\*(?!/))*/\\*" >> [depth](auto& m) { ++(*depth); },
        "(?:[^/]|/(?!\\*))*\\*/" >>
          [depth](auto& m) {
            if (--(*depth) == 0)
              m.mode("start");
          },
      });

    return p;
  }
}
