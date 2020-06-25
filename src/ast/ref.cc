// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ref.h"

using namespace peg::udl;

namespace
{
  void resolve(ast::Ast& ast, err::Errors& err)
  {
    assert(ast->tag == "ref"_);
    auto def = ast::get_def(ast, ast->token);

    if (!def)
    {
      ast::rename(ast, "op");
    }
    else
    {
      switch (def->tag)
      {
        case "classdef"_:
        case "typedef"_:
        case "typeparam"_:
        {
          ast::rename(ast, "typeref");
          break;
        }

        case "field"_:
        case "function"_:
        {
          // could be a member ref if implicit self access is allowed
          ast::rename(ast, "op");
          break;
        }

        case "namedparam"_:
        case "local"_:
        {
          // TODO: use before def
          ast::rename(ast, "localref");
          break;
        }

        default:
        {
          assert(0);
        }
      }
    }
  }
}

namespace ref
{
  void build(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "ref"_:
      {
        resolve(ast, err);
        break;
      }
    }

    ast::for_each(ast, build, err);
  }
}
