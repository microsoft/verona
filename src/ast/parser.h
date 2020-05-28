// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ast.h"

namespace parser
{
  std::string format_error_message(
    const std::string& path, size_t ln, size_t col, const std::string& msg);
  peg::parser create(const std::string& file);
  ast::Ast parse(peg::parser& parser, const std::string& file);

  template<typename T>
  peg::parser create(const T& grammar, const std::string& file)
  {
    peg::parser parser;

    parser.log = [&](size_t ln, size_t col, const std::string& msg) {
      std::cerr << format_error_message(file, ln, col, msg) << std::endl;
    };

    if (!parser.load_grammar(grammar.data(), grammar.size()))
      exit(-1);

    parser.enable_ast<ast::AstImpl>();
    return parser;
  }

  template<typename T>
  ast::Ast parse(peg::parser& parser, const T& src, const std::string& file)
  {
    ast::Ast ast;

    parser.log = [&](size_t ln, size_t col, const std::string& msg) {
      std::cerr << format_error_message(file, ln, col, msg) << std::endl;
    };

    if (!parser.parse_n(src.data(), src.size(), ast, file.c_str()))
      return nullptr;

    return ast;
  }
}
