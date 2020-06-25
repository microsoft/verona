// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"
#include "err.h"

namespace parser
{
  peg::parser create(const std::string& file, err::Errors& err);

  ast::Ast
  parse(peg::parser& parser, const std::string& file, err::Errors& err);

  ast::Ast parse(
    peg::parser& parser,
    const std::string& path,
    const std::string& ext,
    err::Errors& err);

  template<typename T>
  peg::parser
  create(const T& grammar, const std::string& file, err::Errors& err)
  {
    peg::parser parser;

    parser.log = [&](size_t ln, size_t col, const std::string& msg) {
      err << file << ":" << ln << ":" << col << ": " << msg << err::end;
    };

    if (!parser.load_grammar(grammar.data(), grammar.size()))
      return {};

    parser.log = nullptr;
    parser.enable_packrat_parsing();
    parser.enable_ast<ast::AstImpl>();
    return parser;
  }

  template<typename T>
  ast::Ast parse(
    peg::parser& parser,
    const T& src,
    const std::string& file,
    err::Errors& err)
  {
    ast::Ast ast;

    parser.log = [&](size_t ln, size_t col, const std::string& msg) {
      err << file << ":" << ln << ":" << col << ": " << msg << err::end;
    };

    if (!parser.parse_n(src.data(), src.size(), ast, file.c_str()))
      return {};

    return ast;
  }
}
