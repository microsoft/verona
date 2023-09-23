// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"
#include "../wf.h"

namespace verona
{
  Node gamma(Node node)
  {
    // TODO: Move vs Copy type.
    // when testing, this may not resolve
    return (node / Ident)->lookup().front() / Type;
  }

  struct Infer
  {
    Btypes& preds;
    Bounds& bounds;
    Node cls;
    Node ret_type;
    Btype bool_type;

    Infer(Node f, Btypes& p, Bounds& b) : preds(p), bounds(b)
    {
      cls = f->parent(Class);
      ret_type = f / Type;
      bool_type = make_btype(booltype());
      do_block(f / Block, ret_type);
    }

    void check_t_sub_bool(Node from, Node type)
    {
      if (type && !subtype(preds, make_btype(type), bool_type, bounds))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch")
            << clone(type)
            << (ErrorMsg ^ "is not a subtype of std::builtin::Bool"));
      }
    }

    void check_bool_sub_t(Node from, Node type)
    {
      if (type && !subtype(preds, bool_type, make_btype(type), bounds))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch, std::builtin::Bool is not a subtype of")
            << clone(type));
      }
    }

    void check(Node from, Node ltype, Node rtype)
    {
      if (
        ltype && rtype &&
        !subtype(preds, make_btype(ltype), make_btype(rtype), bounds))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch")
            << clone(ltype) << (ErrorMsg ^ "is not a subtype of")
            << clone(rtype));
      }
    }

    void do_block(Node& block, Node block_type)
    {
      for (auto stmt : *block)
      {
        if (stmt == Bind)
        {
          auto rhs = stmt / Rhs;
          auto type = stmt / Type;

          if (rhs == TypeTest)
            check_bool_sub_t(stmt, type);
          else if (rhs == Cast)
            check(stmt, rhs / Type, type);
          else if (rhs->in({Move, Copy}))
            check(stmt, gamma(rhs), type);
          else if (rhs == FieldRef)
          {
            // TODO: Self substitution?
            // may not need this if we set the return type of autofields
            // functions correctly
            check(
              stmt,
              TypeView << -gamma(rhs / Ref)
                       << reftype(
                            cls->lookdown((rhs / Ident)->location()).front() /
                            Type),
              type);
          }
          else if (rhs == Conditional)
          {
            auto cond = rhs / If;
            check_t_sub_bool(cond, gamma(cond));
            do_block(rhs / True, type);
            do_block(rhs / False, type);
          }
          else if (rhs == Call)
          {
            auto sel = rhs / Selector;

            if (sel == FQFunction)
            {
              auto l = resolve_fq(sel);

              if (!l.def)
              {
                block->replace(stmt, err(stmt, "too many arguments"));
                continue;
              }

              auto params = clone(l.def / Params);
              auto ret = clone(l.def / Type);
              auto args = rhs / Args;
              auto arg = args->begin();
              assert(params->size() == args->size());

              l.sub(params);
              l.sub(ret);

              std::for_each(params->begin(), params->end(), [&](Node& param) {
                check(stmt, gamma(*arg++), param / Type);
              });

              check(stmt, ret, type);
            }
            else
            {
              // TODO: selector
            }
          }
          else if (rhs->in(
                     {Int,
                      Bin,
                      Oct,
                      Hex,
                      Float,
                      HexFloat,
                      Char,
                      Escaped,
                      String,
                      Tuple,
                      LLVM}))
          {
            // TODO:
          }
          else
          {
            assert(false);
          }
        }
        else if (stmt == Return)
        {
          assert(stmt == block->back());
          check(stmt, gamma(stmt / Ref), ret_type);
        }
        else if (stmt == Move)
        {
          assert(stmt == block->back());
          check(stmt, gamma(stmt), block_type);
        }
        else
        {
          assert(stmt->in({Class, TypeAlias, LLVM, Drop}));
        }
      }
    }
  };

  PassDef typeinfer()
  {
    auto preds = std::make_shared<Btypes>();
    auto bounds = std::make_shared<Bounds>();

    PassDef pass = {
      "typeinfer",
      wfPassDrop,
      dir::bottomup | dir::once,
      {
        T(Class, TypeAlias) >> ([=](Match&) -> Node {
          preds->pop_back();
          return NoChange;
        }),

        T(Function)[Function] >> ([=](Match& _) -> Node {
          Infer infer(_(Function), *preds, *bounds);
          preds->pop_back();
          return NoChange;
        }),
      }};

    pass.pre(Class, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.pre(TypeAlias, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.pre(Function, [=](Node n) {
      preds->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.post([=](Node) {
      // // TODO: resolve types instead of just printing bounds
      // for (auto& [typevar, bound] : *bounds)
      // {
      //   std::cout << typevar.view() << std::endl << "--lower--" << std::endl;

      //   for (auto& b : bound.lower)
      //     std::cout << b << std::endl;

      //   std::cout << "--upper--" << std::endl;

      //   for (auto& b : bound.upper)
      //     std::cout << b << std::endl;

      //   std::cout << "--done--" << std::endl;
      // }

      return 0;
    });

    return pass;
  }
}
