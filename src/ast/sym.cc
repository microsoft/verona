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
          // could be a mamber ref if implicit self access is allowed
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

  void add_typeargs(ast::Ast& ast, ast::Ast& typeargs)
  {
    // Adds a typeargs node to ast and moves to the next node in the expr.
    // If it isn't a typeargs node, adds an empty typeargs onde and does not
    // advance to the next node in the expr.
    if (typeargs && (typeargs->tag == "typeargs"_))
    {
      auto next = ast::get_next_in_expr(typeargs);
      ast::remove(typeargs);
      ast::push_back(ast, typeargs);
      typeargs = next;
    }
    else
    {
      auto emptyargs = ast::node(ast, "typeargs");
      ast::push_back(ast, emptyargs);
    }
  }

  template<typename T>
  void for_each(ast::Ast ast, err::Errors& err, T f)
  {
    for (decltype(ast->nodes.size()) i = 0; i < ast->nodes.size(); i++)
    {
      auto node = ast->nodes[i];
      f(node, err);
    }
  }

  void elide(ast::Ast& ast, err::Errors& err)
  {
    assert(ast->nodes.size() == 1);
    auto child = ast->nodes[0];
    ast::replace(ast, child);
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

    if (!expr || (expr->nodes[0] != ast))
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
            ast::token(ast->nodes[0], "id", "apply"));
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

      case "term"_:
      {
        ast::rename(ast, "expr");
        sym::scope(ast, err);
        return;
      }

      case "blockexpr"_:
      {
        elide(ast, err);
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
        elide(ast, err);

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

        for_each(ast, err, [](auto& node, auto& err) {
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
            auto id = next->nodes[0];

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
        else
        {
          ast::rename(ast, "op");
        }
        break;
      }
    }

    for_each(ast, err, scope);
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

    for_each(ast, err, references);
  }

  void precedence(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "typeref"_:
      {
        // static-call <-
        //  typeref typargs? (lookup-typeref typeargs?)*
        //  (lookup typeargs?)? tuple?
        // (static-call qualtype function typeargs (args ...))
        auto typeref = ast;
        auto call = ast::node(ast, "static-call");
        ast::replace(ast, call);

        auto qualtype = ast::node(ast, "qualtype");
        ast::push_back(qualtype, typeref);
        ast::push_back(ast, qualtype);

        auto next = ast::get_next_in_expr(ast);
        add_typeargs(qualtype, next);

        // look for type lookups followed by optional typeargs
        auto def = ast::get_def(ast, typeref->token);
        assert(def && (def->tag == "typedef"_));

        while (next)
        {
          if (next->tag != "lookup"_)
            break;

          auto subdef = ast::get_def(def, next->token);

          if (!subdef || (subdef->tag != "typedef"_))
            break;

          ast::remove(next);
          ast::rename(next, "typeref");
          ast::push_back(qualtype, next);
          next = ast::get_next_in_expr(ast);
          add_typeargs(qualtype, next);
          def = subdef;
        }

        if (next && (next->tag == "lookup"_))
        {
          // function call
          ast::remove(next);
          ast::rename(next, "function");
          ast::push_back(call, next);
          next = ast::get_next_in_expr(ast);
          add_typeargs(call, next);
        }
        else
        {
          // create sugar
          auto create = ast::token(call, "function", "create");
          ast::push_back(call, create);
          auto typeargs = ast::node(ast, "typeargs");
          ast::push_back(call, typeargs);
        }

        if (next && (next->tag == "tuple"_))
        {
          ast::remove(next);
          ast::rename(next, "args");
          ast::push_back(call, next);
        }
        else
        {
          auto args = ast::node(call, "args");
          ast::push_back(call, args);
        }
        return;
      }
    }

    for_each(ast, err, precedence);
  }
}
