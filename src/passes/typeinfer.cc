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
    auto l = (node / Ident)->lookup();

    // When testing, this may not resolve, so we just return a typevar.
    if (l.empty())
      return typevar(node);

    return l.front() / Type;
  }

  struct Check
  {
    Node type;
    Btype btype;

    template<typename T>
    Check(T type_) : type(type_), btype(make_btype(type))
    {}

    Check(Btype source, Node& type_) : type(type_), btype(source->make(type)) {}

    operator bool() const
    {
      return bool(type);
    }
  };

  struct Infer
  {
    Btypes& assume;
    SequentPtrs& delayed;
    Node cls;
    Node ret_type;
    Btype bool_type;

    Infer(Node f, Btypes& a, SequentPtrs& d) : assume(a), delayed(d)
    {
      cls = f->parent(Class);
      ret_type = f / Type;
      bool_type = make_btype(booltype());
      do_block(f / Block, ret_type);
    }

    void check_t_sub_bool(Node from, Check type)
    {
      if (type && !subtype(assume, type.btype, bool_type, delayed))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch")
            << clone(type.type)
            << (ErrorMsg ^ "is not a subtype of std::builtin::Bool"));
      }
    }

    void check_bool_sub_t(Node from, Check type)
    {
      if (type && !subtype(assume, bool_type, type.btype, delayed))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch, std::builtin::Bool is not a subtype of")
            << clone(type.type));
      }
    }

    void check(Node from, Check ltype, Check rtype)
    {
      if (ltype && rtype && !subtype(assume, ltype.btype, rtype.btype, delayed))
      {
        from->parent()->replace(
          from,
          err(from, "type mismatch")
            << clone(ltype.type) << (ErrorMsg ^ "is not a subtype of")
            << clone(rtype.type));
      }
    }

    void do_block(Node& block, Node block_type)
    {
      for (auto stmt : *block)
      {
        if (stmt == Bind)
        {
          auto type = stmt / Type;
          auto rhs = stmt / Rhs;

          if (rhs == TypeTest)
            type = Type << booltype(); // FIXED
          else if (rhs == Cast)
            type = clone(rhs / Type); // FIXED
          else if (rhs->in({Move, Copy}))
            type = clone(gamma(rhs)); // FIXED
          else if (rhs == FieldRef)
          {
            // TODO: Self substitution?
            // may not need this if we set the return type of autofields
            // functions correctly
            type = Type
              << (TypeView
                  << clone(gamma(rhs / Ref) / Type)
                  << reftype(clone(
                       cls->lookdown((rhs / Ident)->location()).front() /
                       Type)));
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
              auto bt = make_btype(sel);

              if (bt != Function)
              {
                // In testing, this may also be a non-existant function.
                block->replace(stmt, err(stmt, "too many arguments"));
                continue;
              }

              // Gather all the predicates we will need to prove.
              auto preds = all_predicates(bt->node);
              Nodes errs;

              for (auto& pred : preds)
              {
                if (!subtype(assume, bt->make(pred), delayed))
                  errs.push_back(clone(pred));
              }

              if (!errs.empty())
              {
                auto e = err(clone(rhs), "function call predicate errors");
                rhs << e;

                for (auto& ee : errs)
                  e << (ErrorMsg ^ "this predicate isn't satisfied") << ee;
              }

              auto params = bt->node / Params;
              auto args = rhs / Args;
              auto arg = args->begin();

              // This will only be false in testing.
              if (params->size() == args->size())
              {
                std::for_each(params->begin(), params->end(), [&](Node& param) {
                  check(*arg, gamma(*arg), {bt, param / Type});
                  arg++;
                });
              }

              // Check the function return type.
              check(sel, {bt, bt->node / Type}, type);
            }
            else
            {
              auto args = rhs / Args;
              assert(!args->empty());
              Node params = TypeTuple;

              std::for_each(args->begin(), args->end(), [&](Node& arg) {
                params << clone(gamma(arg));
              });

              // Check the receiver against the selector.
              auto infer = InferSelector << clone(sel) << params << clone(type);
              check(sel, gamma(args->front()), infer);
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
    auto assume = std::make_shared<Btypes>();
    auto delayed = std::make_shared<SequentPtrs>();

    // Block: Bind, Return, Move, Drop, TypeAlias, Class, LLVM.
    // Bind: Copy, Move, TypeTest, Cast.
    //   To do: FieldRef, Call, Literal, Tuple, LLVM.

    PassDef pass = {
      "typeinfer",
      wfPassTypeInfer,
      dir::bottomup | dir::once,
      {
        T(Bind)[Bind] << (T(Ident) * T(Type) * T(TypeTest)) >>
          ([](Match& _) -> Node {
            // Alter in place to maintain symbol table.
            _(Bind) / Type = booltype();
            return NoChange;
          }),

        T(Bind)
            << (T(Ident) * T(Type) *
                (T(Cast) << (T(Copy, Move) * T(Type)[Type]))) >>
          ([](Match& _) -> Node {
            // Alter in place to maintain symbol table.
            _(Bind) / Type = clone(_(Type));
            return NoChange;
          }),

        T(Bind) << (T(Ident) * T(Type) * (T(Move, Copy))[Move]) >>
          ([](Match& _) -> Node {
            // Alter in place to maintain symbol table.
            _(Bind) / Type = clone(gamma(_(Move)));
            return NoChange;
          }),

        T(Bind)
            << (T(Ident)[Ident] * T(Type) *
                (T(Conditional)
                 << (T(Move, Copy)[If] * T(Block)[True] * T(Block)[False]))) >>
          ([](Match& _) -> Node {
            // TODO: gamma(_(If)) <: bool
            // TODO: type = True | False
            return NoChange;
          }),

        T(Return) << T(Move)[Move] >> ([](Match& _) -> Node {
          // TODO: gamma(_(Move)) <: ret_type
          return NoChange;
        }),

        In(Block) * T(Move)[Move] >> ([](Match& _) -> Node {
          // TODO: gamma(_(Move)) <: block_type
          return NoChange;
        }),

        In(Function) * T(Block) >> ([=](Match& _) -> Node {
          // TODO: block_type <: ret_type
          return NoChange;
        }),

        T(Call) << (T(FQFunction)[FQFunction] * T(Args)[Args]) >>
          ([](Match& _) -> Node {
            // TODO: Lookup FQFunction.
            // Prove all predicates. This may involve inferring type arguments.
            // Check all arguments against the function parameters. This may
            // also involve inferring type arguments. Check the function return
            // type against the call site.
            return NoChange;
          }),

        // T(Function)[Function] >> ([=](Match& _) -> Node {
        //   // Our local predicate has already been popped. Add it back.
        //   auto f = _(Function);
        //   assume->push_back(make_btype(f / TypePred));
        //   Infer infer(f, *assume, *delayed);
        //   assume->pop_back();
        //   return NoChange;
        // }),
      }};

    pass.pre({Class, TypeAlias, Function}, [=](Node n) {
      assume->push_back(make_btype(n / TypePred));
      return 0;
    });

    pass.post({Class, TypeAlias, Function}, [=](Node) {
      assume->pop_back();
      return 0;
    });

    pass.post([=](Node) {
      if (!infer_types(*delayed))
      {
        // TODO:
      }

      return 0;
    });

    return pass;
  }
}
