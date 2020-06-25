// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"
#include "dfs.h"
#include "err.h"
#include "parser.h"

namespace module
{
  using Pass = void (*)(ast::Ast& ast, err::Errors& err);
  using Passes = std::vector<Pass>;

  struct Module
  {
    std::string name;
    ast::Ast ast;

    dfs::Color color;
    std::vector<std::shared_ptr<Module>> edges;

    Module(const std::string& name) : name(name), color(dfs::white) {}
  };

  using ModulePtr = std::shared_ptr<Module>;

  ModulePtr build(
    peg::parser& parser,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err);

  ModulePtr build(
    const std::string& grammar,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err);

  inline std::ostream& operator<<(std::ostream& out, ModulePtr m)
  {
    dfs::post(
      m,
      [](auto& m, auto& out) {
        if (m->ast)
          out << peg::ast_to_s(m->ast);
        return true;
      },
      out);

    return out;
  }
}
