// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "sym.h"

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

  void set_fixity(ast::Ast& ast)
  {
    auto prev = ast::get_prev_in_expr(ast);

    if (!prev || (prev->tag == "infix"_))
    {
      // If it's the first atom, or it's after an infix, it's a prefix operator.
      ast::replace(ast, ast::from(ast, "prefix", ast->token));
    }
    else
    {
      // If it's after anything else, it's an infix operator.
      ast::replace(ast, ast::from(ast, "infix", ast->token));
    }
  }

  template<typename T>
  void for_each(ast::Ast ast, err::Errors& err, T f)
  {
    auto size = ast->nodes.size();

    for (decltype(size) i = 0; i < size; i++)
    {
      auto node = ast->nodes[i];
      f(node, err);

      // Back up if we remove nodes. Only remove earlier siblings.
      if (ast->nodes.size() < size)
      {
        auto diff = size - ast->nodes.size();
        i -= diff;
        size = ast->nodes.size();
      }
    }
  }

  void elide_node(ast::Ast& ast, err::Errors& err)
  {
    assert(ast->nodes.size() == 1);
    auto child = ast->nodes[0];
    ast::replace(ast, child);
    sym::build(ast, err);
  }

  void only_atom(ast::Ast& ast, err::Errors& err)
  {
    if (ast::get_expr(ast)->nodes.size() > 1)
    {
      err << ast << ast->name << " must be the only element of an expression."
          << err::end;
    }
  }

  void first_atom(ast::Ast& ast, err::Errors& err)
  {
    if (ast::get_expr(ast)->nodes[0] != ast)
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
      auto seq = block->nodes[0];
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
  void build(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "module"_:
      case "typebody"_:
      case "lambda"_:
      {
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
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "field"_:
      {
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "function"_:
      {
        // A missing function name is sugar for "apply"
        if (ast->nodes[0]->nodes.size() == 0)
        {
          ast->nodes[0]->nodes.push_back(
            ast::from(ast->nodes[0], "id", "apply"));
        }

        auto node = ast->nodes[0]->nodes[0];
        add_symbol(node->token, ast, err);
        add_scope(ast);
        break;
      }

      case "namedparam"_:
      {
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "blockexpr"_:
      {
        elide_node(ast, err);
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
        auto node = ast->nodes[0];

        if (node->tag == "id"_)
          ast::replace(node, ast::from(node, "ref", node->token));

        elide_node(ast, err);
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

        for_each(ast, err, [](auto& node, auto& err) {
          if (node->tag == "id"_)
          {
            ast::replace(node, ast::from(node, "local", node->token));
            add_symbol(node->token, node, err);
          }
        });
        break;
      }

      case "ref"_:
      {
        auto def = ast::get_def(ast, ast->token);

        if (!def || ((def->tag != "local"_) && (def->tag != "namedparam"_)))
          set_fixity(ast);
        break;
      }

      case "sym"_:
      {
        if (ast->token == ".")
        {
          auto next = ast::get_next_in_expr(ast);

          if (next && (next->tag == "atom"_))
          {
            auto lookup = next->nodes[0];

            if ((lookup->tag == "id"_) || (lookup->tag == "sym"_))
            {
              ast::replace(next, ast::from(lookup, "lookup", lookup->token));
              ast::remove(ast);
              break;
            }
          }

          err << ast << "lookup must be followed by a symbol or an identifier."
              << err::end;
        }
        else
        {
          set_fixity(ast);
        }
        break;
      }
    }

    for_each(ast, err, build);
  }
}
