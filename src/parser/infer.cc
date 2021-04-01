// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "infer.h"

#include "ident.h"
#include "lookup.h"
#include "rewrite.h"

namespace verona::parser::infer
{
  struct Infer : Pass<Infer>
  {
    AST_PASS;

    enum class TypeKind
    {
      Base,
      Tuple,
      Member,
      Function,
      TypeVar
    };

    struct TypeInfer
    {
      virtual TypeKind kind() = 0;
    };

    using TypePtr = std::shared_ptr<TypeInfer>;

    struct TypeBase : TypeInfer
    {
      Node<Type> type;

      TypeBase(Node<Type>& type) : type(type) {}

      TypeKind kind()
      {
        return TypeKind::Base;
      }
    };

    struct TypeVar : TypeInfer
    {
      // This is a constrained type variable.
      // TODO: could store program point here and get flow typing
      // vector[(index, Type)]
      std::vector<TypePtr> lower;
      std::vector<TypePtr> upper;

      TypeKind kind()
      {
        return TypeKind::TypeVar;
      }

      void add_lower(TypePtr type)
      {
        lower.push_back(type);
      }

      void add_upper(TypePtr type)
      {
        upper.push_back(type);
      }

      void add_lower(Node<Type>& type)
      {
        if (type)
          lower.push_back(std::make_shared<TypeBase>(type));
      }

      void add_upper(Node<Type>& type)
      {
        if (type)
          upper.push_back(std::make_shared<TypeBase>(type));
      }
    };

    using TypeVarPtr = std::shared_ptr<TypeVar>;

    struct TypeTuple : TypeInfer
    {
      std::vector<TypeVarPtr> list;

      TypeKind kind()
      {
        return TypeKind::Tuple;
      }
    };

    struct TypeMember : TypeInfer
    {
      // This type must have a member with this name and type.
      Location name;
      TypePtr type;

      TypeKind kind()
      {
        return TypeKind::Member;
      }
    };

    struct TypeFunction : TypeInfer
    {
      TypePtr lhs;
      TypePtr rhs;

      TypeKind kind()
      {
        return TypeKind::Function;
      }
    };

    struct Local
    {
      bool assigned;
      bool reassign;
      TypeVarPtr t;
    };

    struct Gamma
    {
      Node<Lambda> lambda;
      std::unordered_map<Location, Local> map;
    };

    std::vector<Gamma> gamma_stack;
    Ident ident;
    Location result = ident("$result");

    Local& g(const Location& name)
    {
      auto& map = gamma_stack.back().map;
      auto find = map.find(name);

      if (find != map.end())
        return find->second;

      auto t = std::make_shared<TypeVar>();
      map.emplace(name, Local{false, false, t});
      return map.at(name);
    }

    Location lhs()
    {
      return parent<Assign>()->left->location;
    }

    void post(Let& let)
    {
      g(let.location).reassign = false;
    }

    void post(Var& var)
    {
      g(var.location).reassign = true;
    }

    void post(Free& fr)
    {
      for (auto it = gamma_stack.rbegin(); it != gamma_stack.rend(); ++it)
      {
        auto find = it->map.find(fr.location);

        if (find != it->map.end())
        {
          if (!find->second.assigned)
          {
            error() << fr.location
                    << "Free variables can't be captured if they haven't been "
                       "assigned to."
                    << text(fr.location) << find->first << "Definition is here."
                    << text(find->first);
          }

          g(fr.location) = find->second;
          return;
        }
      }
    }

    void post(Ref& ref)
    {
      if (parent()->kind() == Kind::Oftype)
        return;

      if (
        (parent()->kind() == Kind::Assign) &&
        (parent<Assign>()->left == current<Expr>()))
      {
        return;
      }

      if (!g(ref.location).assigned)
      {
        error() << ref.location << "Variable used before assignment"
                << text(ref.location);
      }
    }

    void post(Oftype& oftype)
    {
      g(oftype.expr->location).t->add_upper(oftype.type);
    }

    void post(Assign& asn)
    {
      auto& l = g(asn.left->location);

      if (!l.assigned || l.reassign)
      {
        l.assigned = true;
      }
      else
      {
        error() << asn.right->location << "This expression can't be assigned"
                << text(asn.right->location) << asn.left->location
                << "This local has already been assigned to"
                << text(asn.left->location);
      }
    }

    void post(Tuple& tuple)
    {
      auto t = std::make_shared<TypeTuple>();

      for (auto& e : tuple.seq)
        t->list.push_back(g(e->location).t);

      g(lhs()).t->add_lower(t);
    }

    // TODO: select, new, lambda, objectliteral, match, when, constant

    void pre(Lambda& lambda)
    {
      // TODO: don't overwrite parent typeparams
      gamma_stack.push_back({current<Lambda>(), {}});

      for (auto& typeparam : lambda.typeparams)
        g(typeparam->location).t->add_upper(typeparam->type);

      for (auto& param : lambda.params)
      {
        auto& x = g(param->location);
        x.assigned = true;
        x.t->add_upper(param->as<Param>().type);
      }

      g(result).t->add_upper(lambda.result);
    }

    void post(Lambda& lambda)
    {
      gamma_stack.pop_back();
    }
  };

  bool run(Ast& ast, std::ostream& out)
  {
    Infer r;
    r.set_error(out);
    return r << ast;
  }

  struct WF : Pass<WF>
  {
    AST_PASS;
  };

  bool wellformed(Ast& ast, std::ostream& out)
  {
    WF wf;
    wf.set_error(out);
    return wf << ast;
  }
}
