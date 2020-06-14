// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "prec.h"

using namespace peg::udl;

namespace
{
  void add_typeargs(ast::Ast& ast, ast::Ast& typeargs)
  {
    // Adds a typeargs node to ast and moves to the next node in the expr.
    // If it isn't a typeargs node, adds an empty typeargs node and does not
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

  void staticcall_rule(ast::Ast& ast, err::Errors& err)
  {
    // static-call <-
    //  typeref typargs? (lookup-typeref typeargs?)*
    //  (lookup typeargs?)? tuple?
    // (static-call qualtype function typeargs (args ...))
    assert(ast->tag == "typeref"_);
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
      ast::push_back(ast, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
    }
    else
    {
      // create sugar
      auto create = ast::token(ast, "function", "create");
      ast::push_back(ast, create);
      auto typeargs = ast::node(ast, "typeargs");
      ast::push_back(ast, typeargs);
    }

    if (next && (next->tag == "tuple"_))
    {
      ast::remove(next);
      ast::rename(next, "args");
      ast::push_back(ast, next);
    }
    else
    {
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);
    }
  }

  void obj_rule(ast::Ast& ast, err::Errors& err)
  {
    if (ast && (ast->tag == "typeref"_))
      staticcall_rule(ast, err);
  }

  void member_rule(ast::Ast& ast, err::Errors& err)
  {
    // member <- obj (lookup / typeargs tuple? / tuple)*
    // lookup can't distinguish types from fields, so invocation is either
    // calling a method or calling `apply` on a field.
    obj_rule(ast, err);
    auto next = ast::get_next_in_expr(ast);

    while (next)
    {
      switch (next->tag)
      {
        case "lookup"_:
        {
          // (member obj lookup)
          auto obj = ast;
          auto member = ast::node(next, "member");
          ast::replace(ast, member);
          ast::push_back(ast, obj);
          ast::remove(next);
          ast::push_back(ast, next);
          next = ast::get_next_in_expr(ast);
          break;
        }

        case "typeargs"_:
        {
          // (invoke obj typeargs args)
          auto obj = ast;
          auto invoke = ast::node(next, "invoke");
          ast::replace(ast, invoke);
          ast::push_back(ast, obj);
          ast::remove(next);
          ast::push_back(ast, next);
          next = ast::get_next_in_expr(ast);

          if (next && (next->tag == "tuple"_))
          {
            ast::remove(next);
            ast::rename(next, "args");
            ast::push_back(ast, next);
          }
          else
          {
            // zero argument invoke
            auto args = ast::node(ast, "args");
            ast::push_back(ast, args);
          }
          break;
        }

        case "tuple"_:
        {
          // (invoke obj typeargs args)
          auto obj = ast;
          auto invoke = ast::node(next, "invoke");
          ast::replace(ast, invoke);
          ast::push_back(ast, obj);
          auto typeargs = ast::node(ast, "typeargs");
          ast::push_back(ast, typeargs);
          ast::remove(next);
          ast::rename(next, "args");
          ast::push_back(ast, next);
          break;
        }

        default:
          return;
      }
    }
  }

  void prefix_rule(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs obj (args))
    if (ast && (ast->tag == "op"_))
    {
      auto op = ast;
      auto call = ast::node(ast, "call");
      ast::replace(ast, call);
      ast::rename(op, "function");
      ast::push_back(ast, op);
      auto next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
      prefix_rule(next, err);

      if (!next)
      {
        err << ast << "Expected an argument to this prefix function."
            << err::end;
        return;
      }

      ast::remove(next);

      if (next->tag == "tuple"_)
      {
        auto obj = next->nodes.front();
        ast::remove(obj);
        ast::push_back(ast, obj);
        ast::rename(next, "args");
        ast::push_back(ast, next);
      }
      else
      {
        ast::push_back(ast, next);
        auto args = ast::node(ast, "args");
        ast::push_back(ast, args);
      }
    }
    else
    {
      member_rule(ast, err);
    }
  }

  void infix_rule(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs lhs (args rhs))
    // Infix operators are left associative.
    prefix_rule(ast, err);
    auto next = ast::get_next_in_expr(ast);
    std::string op;

    while (next)
    {
      if (next->tag == "assign"_)
        return;

      if (next->tag != "op"_)
      {
        err << next << "Expected an infix operator here." << err::end;
        return;
      }

      if (op.empty())
      {
        op = next->token;
      }
      else if (op != next->token)
      {
        err << next << "Use parentheses to establish precedence between " << op
            << " and " << next->token << " infix operators." << err::end;
      }

      auto lhs = ast;
      auto call = ast::node(ast, "call");
      ast::replace(ast, call);
      ast::remove(next);
      ast::rename(next, "function");
      ast::push_back(ast, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
      ast::push_back(ast, lhs);
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);

      if (!next)
      {
        err << ast << "Expected an expression after this infix operator."
            << err::end;
        return;
      }

      prefix_rule(next, err);

      if (next)
      {
        ast::remove(next);
        ast::push_back(args, next);
        next = ast::get_next_in_expr(ast);
      }
    }
  }

  void assign_rule(ast::Ast& ast, err::Errors& err)
  {
    // (assign lhs rhs)
    // Assignment is right associative.
    infix_rule(ast, err);
    auto next = ast::get_next_in_expr(ast);

    if (!next)
      return;

    if (next->tag != "assign"_)
    {
      err << next << "Expected an assignment or the end of an expression here."
          << err::end;
      return;
    }

    auto lhs = ast;
    auto assign = ast::node(ast, "assign");
    ast::replace(ast, assign);
    ast::push_back(ast, lhs);
    ast::remove(next);
    next = ast::get_next_in_expr(ast);
    assign_rule(next, err);

    if (!next)
    {
      err << ast << "Expected an expression after this assignment." << err::end;
      return;
    }

    ast::remove(next);
    ast::push_back(ast, next);
  }
}

namespace prec
{
  void build(ast::Ast& ast, err::Errors& err)
  {
    // static-call <-
    //  typeref typargs? (lookup-typeref typeargs?)* (lookup typeargs?)? tuple?
    // obj <-
    //  let / localref / tuple / new / lambda / literal / blockexpr /
    //  static-call
    // member <- obj (lookup / typeargs tuple? / tuple)*
    // prefix-call <- (op typeargs? prefix-call) / member
    // infix-call <- prefix-call (op typeargs? prefix-call)*
    // assign <- infix-call ('=' assign)?
    // expr <- break / continue / return assign? / yield assign? / assign
    switch (ast->tag)
    {
      case "expr"_:
      {
        if (ast->nodes.empty())
        {
          ast::remove(ast);
          return;
        }

        auto expr = ast;
        ast = ast->nodes.front();

        switch (ast->tag)
        {
          case "break"_:
          case "continue"_:
            break;

          case "return"_:
          case "yield"_:
          {
            auto control = ast::node(ast, ast->name.c_str());
            ast::replace(ast, control);
            auto next = ast::get_next_in_expr(ast);
            assign_rule(next, err);
            ast::remove(next);
            ast::push_back(ast, next);
            break;
          }

          default:
          {
            assign_rule(ast, err);
            break;
          }
        }

        auto parent = expr->parent.lock();

        if (parent && (parent->tag == "interp_string"_))
        {
          // (call (function 'string') (typeargs) obj (args))
          auto call = ast::node(expr, "call");
          auto fun = ast::token(expr, "function", "string");
          ast::push_back(call, fun);
          auto typeargs = ast::node(expr, "typeargs");
          ast::push_back(call, typeargs);
          auto obj = expr->nodes.front();
          ast::remove(obj);
          ast::push_back(call, obj);
          auto args = ast::node(expr, "args");
          ast::push_back(call, args);
          ast::push_back(expr, call);
        }

        if (expr->nodes.size() == 1)
        {
          ast::elide(expr);
          build(expr, err);
          return;
        }
        break;
      }

      case "tuple"_:
      {
        if (ast->nodes.size() == 1)
        {
          ast::elide(ast);
          build(ast, err);
          return;
        }
        break;
      }
    }

    ast::for_each(ast, build, err);
  }
}
