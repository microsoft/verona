// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  void extract_typeparams(Node scope, Node t, Node tp)
  {
    // This function extracts all typeparams from a type `t` that are defined
    // within `scope` and appends them to `tp` if they aren't already present.
    if (t->type().in(
          {Type,
           TypeArgs,
           TypeUnion,
           TypeIsect,
           TypeTuple,
           TypeList,
           TypeView}))
    {
      for (auto& tt : *t)
        extract_typeparams(scope, tt, tp);
    }
    else if (t->type().in({TypeClassName, TypeAliasName, TypeTraitName}))
    {
      extract_typeparams(scope, t / Lhs, tp);
      extract_typeparams(scope, t / TypeArgs, tp);
    }
    else if (t->type() == TypeParamName)
    {
      auto id = t / Ident;
      auto defs = id->lookup(scope);

      if ((defs.size() == 1) && (defs.front()->type() == TypeParam))
      {
        if (!std::any_of(tp->begin(), tp->end(), [&](auto& p) {
              return (p / Ident)->location() == id->location();
            }))
        {
          tp << clone(defs.front());
        }
      }

      extract_typeparams(scope, t / Lhs, tp);
      extract_typeparams(scope, t / TypeArgs, tp);
    }
  }

  Node typeparams_to_typeargs(Node node, Node typeargs = TypeArgs)
  {
    // This finds all typeparams in a Class or Function definition and builds
    // a TypeArgs that contains all of them, in order.
    if (!node->type().in({Class, Function}))
      return typeargs;

    for (auto typeparam : *(node / TypeParams))
    {
      typeargs
        << (Type
            << (TypeParamName << DontCare << clone(typeparam / Ident)
                              << TypeArgs));
    }

    return typeargs;
  }

  PassDef partialapp()
  {
    // This should happen after `lambda` (so that anonymous types get partial
    // application), after `autocreate` (so that constructors get partial
    // application), and after `defaultargs` (so that default arguments don't
    // get partial application).

    // This means that partial application can't be written in terms of lambdas,
    // but instead has to be anonymous classes. There's no need to check for
    // non-local returns.
    return {
      dir::bottomup | dir::once,
      {
        T(Function)[Function]
            << ((T(Ref) / T(DontCare))[Ref] * Name[Id] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type) *
                T(DontCare) * T(TypePred)[TypePred] *
                (T(Block) / T(DontCare))) >>
          [](Match& _) {
            // Create a FunctionName for a static call to the original function.
            auto f = _(Function);
            auto id = _(Id);
            auto parent = f->parent()->parent()->shared_from_this();

            auto func_name = FunctionName
              << (((parent->type() == Class) ? TypeClassName : TypeTraitName)
                  << DontCare << clone(parent / Ident)
                  << typeparams_to_typeargs(parent))
              << clone(id) << typeparams_to_typeargs(f);

            // Find the lowest arity that is not already defined. If an arity 5
            // and an arity 3 function `f` are provided, an arity 4 partial
            // application will be generated that calls the arity 5 function,
            // and arity 0-2 functions will be generated that call the arity 3
            // function.
            auto defs = parent->lookdown(id->location());
            auto params = _(Params);
            size_t start_arity = 0;
            auto end_arity = params->size();

            for (auto def : defs)
            {
              if ((def == f) || (def->type() != Function))
                continue;

              auto arity = (def / Params)->size();

              if (arity < end_arity)
                start_arity = std::max(start_arity, arity + 1);
            }

            // Create a unique anonymous class name for each arity.
            Nodes names;

            for (auto arity = start_arity; arity < end_arity; ++arity)
              names.push_back(Ident ^ _.fresh(l_class));

            Node ret = Seq;
            auto ref = _(Ref);
            Node call = (ref->type() == Ref) ? CallLHS : Call;

            for (auto arity = start_arity; arity < end_arity; ++arity)
            {
              // Create an anonymous class for each arity.
              auto name = names[arity - start_arity];
              Node class_tp = TypeParams;
              Node classbody = ClassBody;
              auto classdef = Class << clone(name) << class_tp << inherit()
                                    << typepred() << classbody;

              // The anonymous class has fields for each supplied argument and a
              // create function that captures the supplied arguments.
              Node create_params = Params;
              Node new_args = Args;
              classbody
                << (Function << DontCare << (Ident ^ create) << TypeParams
                             << create_params << typevar(_) << DontCare
                             << typepred()
                             << (Block << (Expr << (Call << New << new_args))));

              // Create a function that returns the anonymous class for each
              // arity.
              Node func_tp = TypeParams;
              Node func_params = Params;
              Node func_args = Args;
              auto func = Function
                << clone(ref) << clone(id) << func_tp << func_params
                << typevar(_) << DontCare << typepred()
                << (Block
                    << (Expr
                        << (Call << (FunctionName
                                     << (TypeClassName
                                         << DontCare << clone(name) << TypeArgs)
                                     << (Ident ^ create) << TypeArgs)
                                 << func_args)));

              for (size_t i = 0; i < arity; ++i)
              {
                auto param = params->at(i);
                auto param_id = param / Ident;
                auto param_type = param / Type;

                extract_typeparams(f, param_type, class_tp);
                extract_typeparams(f, param_type, func_tp);

                classbody << (FieldLet << clone(param_id) << clone(param_type));
                create_params << clone(param);
                new_args << (Expr << (RefLet << clone(param_id)));
                func_params << clone(param);
                func_args << (Expr << (RefLet << clone(param_id)));
              }

              // The anonymous class has a function for each intermediate arity
              // and for the final arity.
              for (auto i = arity + 1; i <= end_arity; ++i)
              {
                // TODO: capability for Self, depends on captured param types
                auto self_id = Ident ^ _.fresh(l_self);
                Node apply_tp = TypeParams;
                Node apply_params = Params
                  << (Param << self_id << (Type << Self));
                Node apply_pred;
                Node fwd_args = Args;

                for (size_t j = 0; j < arity; ++j)
                {
                  // Include our captured arguments.
                  fwd_args
                    << (Expr
                        << (Call
                            << (Selector << clone(params->at(j) / Ident)
                                         << TypeArgs)
                            << (Args << (Expr << (RefLet << clone(self_id))))));
                }

                for (auto j = arity; j < i; ++j)
                {
                  // Add the additional arguments passed to this apply function.
                  auto param = params->at(j);
                  extract_typeparams(f, param / Type, apply_tp);
                  apply_params << clone(param);
                  fwd_args << (Expr << (RefLet << clone(param / Ident)));
                }

                Node fwd;

                if (i == end_arity)
                {
                  // The final arity calls the original function. It has the
                  // type predicate from the original function.
                  apply_pred = clone(_(TypePred));
                  fwd = clone(func_name);
                }
                else
                {
                  // Intermediate arities call the next arity. No type predicate
                  // is applied.
                  apply_pred = typepred();
                  fwd = FunctionName
                    << (TypeClassName
                        << DontCare << clone(names[i - start_arity])
                        << typeparams_to_typeargs(
                             apply_tp, typeparams_to_typeargs(class_tp)))
                    << (Ident ^ create) << TypeArgs;
                }

                classbody
                  << (Function
                      << clone(ref) << apply_id() << apply_tp << apply_params
                      << typevar(_) << DontCare << apply_pred
                      << (Block << (Expr << (clone(call) << fwd << fwd_args))));
              }

              ret << classdef << func;
            }

            return ret << f;
          },
      }};
  }
}
