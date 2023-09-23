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
    Btype ret_type;
    Btype bool_type;

    Infer(Node f, Btypes& p, Bounds& b) : preds(p), bounds(b)
    {
      cls = f->parent(Class);
      ret_type = make_btype(f / Type);
      bool_type = make_btype(booltype());
      check(do_block(f / Block), ret_type);
    }

    void check(Btype ltype, Btype rtype)
    {
      if (!subtype(preds, ltype, rtype, bounds))
      {
        // TODO:
        subtype(preds, ltype, rtype, bounds);
        std::cout << "subtype failed" << std::endl
                  << "---" << std::endl
                  << ltype << "---" << std::endl
                  << rtype;
      }
    }

    void check(Node ltype, Btype rtype)
    {
      if (ltype)
        check(make_btype(ltype), rtype);
    }

    Node do_block(Node& block)
    {
      for (auto stmt : *block)
      {
        if (stmt == Bind)
        {
          // TODO: literals
          auto rhs = stmt / Rhs;
          auto type = make_btype(stmt / Type);

          if (rhs == TypeTest)
            check(bool_type, type);
          else if (rhs == Cast)
            check(rhs / Type, type);
          else if (rhs->in({Move, Copy}))
            check(gamma(rhs), type);
          else if (rhs == FieldRef)
          {
            // TODO: Self substitution?
            // this is failing when `type` is a TypeVar, unclear why
            check(
              TypeView << -gamma(rhs / Ref)
                       << reftype(
                            cls->lookdown((rhs / Ident)->location()).front() /
                            Type),
              type);
          }
          else if (rhs == Conditional)
          {
            check(gamma(rhs / If), bool_type);
            check(do_block(rhs / True), type);
            check(do_block(rhs / False), type);
          }
          else if (rhs == Call)
          {
            auto sel = rhs / Selector;

            if (sel == FQFunction)
            {
              // TODO: may not have a function of this arity
              auto l = resolve_fq(sel);
              auto params = clone(l.def / Params);
              auto ret = clone(l.def / Type);
              auto args = rhs / Args;
              l.sub(params);
              l.sub(ret);

              (void)std::equal(
                params->begin(),
                params->end(),
                args->begin(),
                args->end(),
                [&](Node& param, Node& arg) {
                  check(gamma(arg), make_btype(param / Type));
                  return true;
                });

              check(ret, type);
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
          check(gamma(stmt / Ref), ret_type);
        }
        else if (stmt == Move)
        {
          assert(stmt == block->back());
          return gamma(stmt);
        }
        else
        {
          assert(stmt->in({Class, TypeAlias, LLVM, Drop}));
        }
      }

      return {};
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
      for (auto& [typevar, bound] : *bounds)
      {
        std::cout << typevar.view() << std::endl << "--lower--" << std::endl;

        for (auto& b : bound.lower)
          std::cout << b << std::endl;

        std::cout << "--upper--" << std::endl;

        for (auto& b : bound.upper)
          std::cout << b << std::endl;

        std::cout << "--done--" << std::endl;
      }

      return 0;
    });

    return pass;
  }
}
