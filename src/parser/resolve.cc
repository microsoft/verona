// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "resolve.h"

#include "ident.h"
#include "lookup.h"
#include "rewrite.h"

namespace verona::parser::resolve
{
  struct Resolve : Pass<Resolve>
  {
    AST_PASS;

    Ident ident;
    Location token_create;

    Resolve()
    {
      token_create = ident("create");
    }

    void create_sugar(StaticRef& sr, Ast& def)
    {
      // We found a type as an expression, so we'll turn it into a constructor.
      auto find = def->get_sym(token_create);
      bool has_params = false;

      // Assume it's a zero argument create unless we can discover otherwise.
      if (find && (find->kind() == Kind::Function))
        has_params = find->as<Function>().signature->params.size() > 0;

      auto create = std::make_shared<TypeName>();
      create->location = token_create;

      auto sref = std::make_shared<StaticRef>();
      sref->location = sr.location;
      sref->typenames = sr.typenames;
      sref->typenames.push_back(create);

      if (has_params)
      {
        // If create takes parameters, leave it as a static function reference.
        // (staticref typename...)
        // ->
        // (staticref typename...::create)
        rewrite(stack, sref);
      }
      else
      {
        // If create takes no parameters, apply it now.
        // (staticref typename...)
        // ->
        // (apply (staticref typename...::create) ())
        auto apply = std::make_shared<Apply>();
        apply->location = sr.location;
        apply->expr = sref;
        rewrite(stack, apply);
      }
    }

    void unpack(Node<Tuple>& tuple, Node<Expr>& expr)
    {
      if (expr->kind() == Kind::Tuple)
      {
        auto& pack = expr->as<Tuple>();

        for (auto& expr : pack.seq)
          tuple->seq.push_back(expr);
      }
      else
      {
        tuple->seq.push_back(expr);
      }
    }

    void post(TypeRef& tr)
    {
      // This checks that the type exists but doesn't rewrite the AST.
      auto paths = look_up(stack, tr.typenames);

      if (paths.empty())
      {
        error() << tr.location << "Couldn't find a definition of this type."
                << text(tr.location);
        return;
      }

      if (paths.size() > 1)
      {
        auto& out = error()
          << tr.location << "Found multiple definitions of this type."
          << text(tr.location);

        for (auto& path : paths)
        {
          auto& loc = path.back()->location;
          out << loc << "Found a definition here." << text(loc);
        }
        return;
      }

      auto& def = paths.front().back();

      if (!is_kind(
            def->kind(),
            {Kind::Class, Kind::Interface, Kind::TypeAlias, Kind::TypeParam}))
      {
        error() << tr.location << "Expected a type, but got a "
                << kindname(def->kind()) << text(tr.location) << def->location
                << "Definition is here" << text(def->location);
      }
    }

    void post(StaticRef& sr)
    {
      auto paths = look_up(stack, sr.typenames);

      if (paths.empty())
      {
        // If it's a single element name in an operator position, it will be
        // treated as a member selection.
        if (sr.typenames.size() == 1)
        {
          auto ast = parent();

          if (ast->kind() == Kind::Infix)
          {
            if (ast->as<Infix>().op == stack.back())
              sr.maybe_member = true;
          }
          else if (ast->kind() == Kind::Prefix)
          {
            if (ast->as<Prefix>().op == stack.back())
              sr.maybe_member = true;
          }
        }

        if (!sr.maybe_member)
        {
          error() << sr.location
                  << "Couldn't find a definition for this reference."
                  << text(sr.location);
        }
        return;
      }

      if (paths.size() > 1)
      {
        auto& out = error()
          << sr.location << "Found multiple definitions of this reference."
          << text(sr.location);

        for (auto& path : paths)
        {
          auto& loc = path.back()->location;
          out << loc << "Found a definition here." << text(loc);
        }
        return;
      }

      auto& def = paths.front().back();

      if (is_kind(def->kind(), {Kind::Class, Kind::Interface, Kind::TypeAlias}))
      {
        create_sugar(sr, def);
      }
      else if (!is_kind(def->kind(), {Kind::Function}))
      {
        error() << sr.location
                << "Expected a type or a static function, but got a "
                << kindname(def->kind()) << text(sr.location) << def->location
                << "Definition is here" << text(def->location);
      }
    }

    void post(Specialise& spec)
    {
      // (specialise staticref typeargs)
      // ->
      // (specialise (apply staticref ()) typeargs)
      if (spec.expr->kind() == Kind::StaticRef)
      {
        auto apply = std::make_shared<Apply>();
        apply->location = spec.expr->location;
        apply->expr = spec.expr;
        spec.expr = apply;
      }
    }

    void post(Select& select)
    {
      if (select.expr->kind() == Kind::StaticRef)
      {
        error() << select.expr->location
                << "Can't select a member on a static function"
                << text(select.expr->location);
      }
    }

    void post(Prefix& prefix)
    {
      auto apply = std::make_shared<Apply>();
      apply->location = prefix.location;

      if (prefix.op->kind() == Kind::StaticRef)
      {
        auto& sr = prefix.op->as<StaticRef>();

        if (sr.maybe_member)
        {
          auto select = std::make_shared<Select>();
          select->location = sr.location;
          select->expr = prefix.expr;
          select->member = sr.typenames.back()->location;

          if (!sr.typenames.back()->typeargs.empty())
          {
            // (prefix maybe-member[T] expr)
            // ->
            // (apply (specialise (select expr member) T) ())
            auto spec = std::make_shared<Specialise>();
            spec->location = sr.location;
            spec->expr = select;
            spec->typeargs = sr.typenames.back()->typeargs;
            apply->expr = spec;
          }
          else
          {
            // (prefix maybe-member expr)
            // ->
            // (apply (select expr member) ())
            apply->expr = select;
          }
        }
      }

      if (!apply->expr)
      {
        // (prefix op expr)
        // ->
        // (apply op expr)
        apply->expr = prefix.op;
        apply->args = prefix.expr;
      }

      rewrite(stack, apply);
    }

    void post(Infix& infix)
    {
      auto apply = std::make_shared<Apply>();
      apply->location = infix.location;

      if (infix.op->kind() == Kind::StaticRef)
      {
        auto& sr = infix.op->as<StaticRef>();

        if (sr.maybe_member)
        {
          auto select = std::make_shared<Select>();
          select->location = sr.location;
          select->expr = infix.left;
          select->member = sr.typenames.back()->location;

          if (!sr.typenames.back()->typeargs.empty())
          {
            // (infix maybe-member[T] expr1 expr2)
            // ->
            // (apply (specialise (select expr1 member) T) expr2)
            auto spec = std::make_shared<Specialise>();
            spec->location = sr.location;
            spec->expr = select;
            spec->typeargs = sr.typenames.back()->typeargs;
            apply->expr = spec;
          }
          else
          {
            // (infix maybe-member expr1 expr2)
            // ->
            // (apply (select expr1 member) expr2)
            apply->expr = select;
          }

          apply->args = infix.right;
        }
      }

      if (!apply->expr)
      {
        // (infix _ expr1 expr2)
        // ->
        // (apply _ (tuple <unpack>expr1 <unpack>expr2))
        auto tuple = std::make_shared<Tuple>();
        tuple->location = infix.left->location;
        unpack(tuple, infix.left);
        unpack(tuple, infix.right);

        apply->expr = infix.op;
        apply->args = tuple;
      }

      rewrite(stack, apply);
    }
  };

  bool run(Ast& ast, std::ostream& out)
  {
    Resolve r;
    r.set_error(out);
    return r << ast;
  }

  struct WF : Pass<WF>
  {
    AST_PASS;

    void post(Prefix& node)
    {
      error() << node.location << "Unexpected " << kindname(node.kind())
              << " after resolve pass." << text(node.location);
    }

    void post(Infix& node)
    {
      error() << node.location << "Unexpected " << kindname(node.kind())
              << " after resolve pass." << text(node.location);
    }
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
