// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "sym.h"

#include "lit.h"

using namespace peg::udl;

namespace
{
  void add_scope(ast::Ast& ast)
  {
    ast->scope = std::make_shared<ast::SymbolScope>();
  }

  void add_symbol(ast::Ident id, ast::Ast& ast, err::Errors& err)
  {
    auto enclosing = ast::get_scope(ast);

    if (!enclosing)
    {
      err << ast << id << " has no enclosing scope." << err::end;
      return;
    }

    auto scope = enclosing->scope;
    auto find = scope->sym.find(id);

    if (find != scope->sym.end())
    {
      auto prev = find->second.lock();
      err << ast << id << " is already defined." << prev
          << "Previous definition of " << id << " is here." << err::end;
    }

    scope->sym.emplace(id, ast);
  }

  void resolve_ref(ast::Ast& ast, err::Errors& err)
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

  void only_atom(ast::Ast& ast, err::Errors& err)
  {
    auto expr = ast::get_expr(ast);

    if (!expr || (expr->nodes.size() > 1))
    {
      err << ast << ast->name << " must be the only element of an expression."
          << err::end;
    }
  }

  void first_atom(ast::Ast& ast, err::Errors& err)
  {
    auto expr = ast::get_expr(ast);

    if (!expr || (expr->nodes.front() != ast))
    {
      err << ast << ast->name << " must be the first element of an expression."
          << err::end;
    }
  }

  void not_in_pattern(ast::Ast& ast, err::Errors& err)
  {
    if (ast::get_closest(ast, "pattern"_))
      err << ast << ast->token << " cannot appear in a pattern." << err::end;
  }

  void last_in_block(ast::Ast& ast, err::Errors& err)
  {
    auto block = ast::get_closest(ast, "block"_);

    if (block)
    {
      auto seq = block->nodes.front();
      auto expr = seq->nodes.back();

      if (expr == ast::get_expr(ast))
        return;
    }

    err << ast << ast->token << " must be the last expression in a block."
        << err::end;
  }
}

namespace sym
{
  void scope(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "module"_:
      case "lambda"_:
      {
        add_scope(ast);
        break;
      }

      case "new"_:
      {
        if (ast->nodes.size() == 0)
          err << ast << "new requires type name or body" << err::end;

        add_scope(ast);
        break;
      }

      case "typedef"_:
      {
        add_symbol(ast->nodes[1]->token, ast, err);
        add_scope(ast);
        not_in_pattern(ast, err);
        break;
      }

      case "typeparam"_:
      {
        add_symbol(ast->nodes.front()->token, ast, err);
        break;
      }

      case "field"_:
      {
        add_symbol(ast->nodes.front()->token, ast, err);
        break;
      }

      case "function"_:
      {
        // A missing function name is sugar for "apply"
        if (ast->nodes.front()->nodes.size() == 0)
        {
          ast->nodes.front()->nodes.push_back(
            ast::token(ast->nodes.front(), "id", "apply"));
        }

        auto node = ast->nodes.front()->nodes.front();
        add_symbol(node->token, ast, err);
        add_scope(ast);
        break;
      }

      case "namedparam"_:
      {
        add_symbol(ast->nodes.front()->token, ast, err);
        break;
      }

      case "term"_:
      {
        ast::rename(ast, "expr");
        scope(ast, err);
        return;
      }

      case "blockexpr"_:
      {
        ast::elide(ast);
        scope(ast, err);
        return;
      }

      case "block"_:
      case "when"_:
      case "while"_:
      case "match"_:
      case "case"_:
      case "if"_:
      case "for"_:
      {
        add_scope(ast);
        not_in_pattern(ast, err);
        break;
      }

      case "atom"_:
      {
        ast::elide(ast);

        if (ast->tag == "id"_)
          ast::rename(ast, "ref");

        scope(ast, err);
        return;
      }

      case "break"_:
      case "continue"_:
      {
        only_atom(ast, err);
        not_in_pattern(ast, err);
        last_in_block(ast, err);
        break;
      }

      case "return"_:
      case "yield"_:
      {
        first_atom(ast, err);
        not_in_pattern(ast, err);
        last_in_block(ast, err);
        break;
      }

      case "let"_:
      {
        first_atom(ast, err);

        ast::for_each(ast, err, [](auto& node, auto& err) {
          if (node->tag == "id"_)
          {
            ast::rename(node, "local");
            add_symbol(node->token, node, err);
          }
        });
        break;
      }

      case "sym"_:
      {
        if (ast->token == ".")
        {
          auto next = ast::get_next_in_expr(ast);

          if (next && (next->tag == "atom"_))
          {
            auto id = next->nodes.front();

            if ((id->tag == "id"_) || (id->tag == "sym"_))
            {
              auto lookup = ast::token(id, "lookup", id->token);
              ast::replace(ast, lookup);
              ast::remove(next);
              break;
            }
          }

          err << ast << "lookup must be followed by a symbol or an identifier."
              << err::end;
        }
        else if (ast->token == "=")
        {
          ast::rename(ast, "assign");
        }
        else
        {
          ast::rename(ast, "op");
        }
        break;
      }

      case "string"_:
      {
        auto s = lit::escape(ast->token);
        auto e = ast::token(ast, "string", s);
        ast::replace(ast, e);
        break;
      }

      case "interp_string"_:
      {
        lit::mangle_indent(ast);
        break;
      }

      case "quote"_:
      {
        // Quote elements of an interpolated string are not escaped.
        ast::rename(ast, "string");
        break;
      }

      case "unquote"_:
      {
        if (ast->nodes.front()->tag == "%word"_)
        {
          auto ref = ast->nodes.front();
          auto expr = ast::node(ast, "expr");
          ast::replace(ast, expr);
          ast::rename(ref, "ref");
          ast::push_back(ast, ref);
        }
        else
        {
          ast::elide(ast);
        }

        scope(ast, err);
        return;
      }
    }

    ast::for_each(ast, err, scope);
  }

  void references(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "ref"_:
      {
        resolve_ref(ast, err);
        break;
      }
    }

    ast::for_each(ast, err, references);
  }
}
