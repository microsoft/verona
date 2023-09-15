// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../subtype.h"

namespace verona
{
  Node gamma(Node node)
  {
    // TODO: Move vs Copy type.
    return (node / Ident)->lookup().front() / Type;
  }

  struct Infer
  {
    Bounds& bounds;
    Node cls;
    Btype ret_type;
    Btype bool_type;

    Infer(Node f, Bounds& b) : bounds(b)
    {
      cls = f->parent(Class);
      ret_type = make_btype(f / Type);
      bool_type = make_btype(booltype());
      check(do_block(f / Block), ret_type);
    }

    void check(Btype ltype, Btype rtype)
    {
      if (!subtype(ltype, rtype, bounds))
      {
        // TODO:
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
              auto l = resolve_fq(sel);
              auto params = clone(l.def / Params);
              auto ret = clone(l.def / Type);
              auto args = rhs / Args;
              l.sub(params);
              l.sub(ret);

              std::equal(
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
    auto bounds = std::make_shared<Bounds>();

    PassDef typeinfer = {
      dir::bottomup | dir::once,
      {
        T(Function)[Function] >> ([=](Match& _) -> Node {
          Infer infer(_(Function), *bounds);
          return NoChange;
        }),
      }};

    typeinfer.post([=](Node) {
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

    return typeinfer;
  }
}
