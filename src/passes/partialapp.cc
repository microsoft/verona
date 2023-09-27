// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../lookup.h"
#include "../wf.h"

namespace verona
{
  void extract_typeparams(Node scope, Node t, Node tp)
  {
    // TODO: use FQNs

    // This function extracts all typeparams from a type `t` that are defined
    // within `scope` and appends them to `tp` if they aren't already present.
    if (t->in(
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
    else if (t->in({TypeClassName, TypeAliasName, TypeTraitName}))
    {
      extract_typeparams(scope, t / Lhs, tp);
      extract_typeparams(scope, t / TypeArgs, tp);
    }
    else if (t == TypeParamName)
    {
      auto id = t / Ident;
      auto defs = id->lookup(scope);

      if ((defs.size() == 1) && ((*defs.begin()) == TypeParam))
      {
        if (!std::any_of(tp->begin(), tp->end(), [&](auto& p) {
              return (p / Ident)->location() == id->location();
            }))
        {
          tp << clone(*defs.begin());
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
    if (!node->in({Class, Function}))
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
      "partialapp",
      wfPassAutoFields,
      dir::bottomup | dir::once,
      {
        T(Function)[Function]
            << (IsImplicit * Hand[Ref] * T(Ident)[Ident] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type) *
                T(DontCare) * T(TypePred)[TypePred] * T(Block, DontCare)) >>
          ([](Match& _) -> Node {
            auto f = _(Function);
            auto parent = f->parent({Class, Trait});
            auto hand = _(Ref)->type();
            auto id = _(Ident);
            auto params = _(Params);

            // Find the lowest arity that is not already defined. If an arity 5
            // and an arity 3 function `f` are provided, an arity 4 partial
            // application will be generated that calls the arity 5 function,
            // and arity 0-2 functions will be generated that call the arity 3
            // function.
            auto defs = parent->lookdown(id->location());
            size_t start_arity = 0;
            auto end_arity = params->size();

            for (auto def : defs)
            {
              if ((def == f) || (def != Function))
                continue;

              auto arity = (def / Params)->size();

              if (arity < end_arity)
                start_arity = std::max(start_arity, arity + 1);
            }

            if (start_arity == end_arity)
              return NoChange;

            // We will be returning the original function, plus some number of
            // partial application functions and their anonymous classes. Make
            // the local FQ before putting `f` into the Seq node.
            auto fq_f = local_fq(f);
            Node ret = Seq << f;

            // If the parent is a trait, generate the partial application
            // function prototypes, but no implementations.
            if (parent == Trait)
            {
              for (auto arity = start_arity; arity < end_arity; ++arity)
              {
                Node func_tp = TypeParams;
                Node func_params = Params;

                for (size_t i = 0; i < arity; ++i)
                {
                  auto param = params->at(i);
                  auto param_id = param / Ident;
                  auto param_type = param / Type;

                  // Add any needed typeparams.
                  extract_typeparams(f, param_type, func_tp);

                  // Add the parameter to the partial function.
                  func_params << clone(param);
                }

                ret
                  << (Function << Implicit << hand << clone(id) << func_tp
                               << func_params << typevar(_) << DontCare
                               << typepred() << DontCare);
              }

              return ret;
            }

            // Create a unique anonymous class name for each arity.
            Nodes names;
            auto basename = std::string("partial.")
                              .append(hand.str())
                              .append(".")
                              .append(id->location().view())
                              .append("/");

            for (auto arity = start_arity; arity < end_arity; ++arity)
            {
              names.push_back(
                Ident ^ _.fresh(basename + std::to_string(arity)));
            }

            auto fq_parent = local_fq(parent);
            Nodes fqs;
            Nodes classbodies;

            for (auto arity = start_arity; arity < end_arity; ++arity)
            {
              // Create an anonymous class for each arity.
              auto name = names[arity - start_arity];
              Node class_tp = TypeParams;
              Node classbody = ClassBody;
              auto classdef = Class << clone(name) << class_tp
                                    << (Inherit << DontCare) << typepred()
                                    << classbody;

              auto fq_class = append_fq(
                fq_parent,
                TypeClassName << clone(name)
                              << typeparams_to_typeargs(classdef));

              // The anonymous class has fields for each supplied argument and a
              // create function that captures the supplied arguments.
              auto fq_new = append_fq(fq_class, selector(l_new));
              Node new_params = Params;
              Node new_args = Tuple;

              // Find all needed typeparams and add them.
              Node func_tp = TypeParams;
              Node func_params = Params;

              Node create_params = Params;
              Node create_args = Tuple;

              for (size_t i = 0; i < arity; ++i)
              {
                auto param = params->at(i);
                auto param_id = param / Ident;
                auto param_type = param / Type;

                // Add any needed typeparams to both the class and the function.
                extract_typeparams(f, param_type, class_tp);
                extract_typeparams(f, param_type, func_tp);

                // Add the field to the anonymous class.
                classbody << (FieldLet << clone(param_id) << clone(param_type));

                // Add the parameter to the create function on the class.
                create_params << clone(param);

                // Add the argument to the `new` call inside the class create
                // function.
                new_args << (Expr << (RefLet << clone(param_id)));
                new_params << (Param << clone(param_id) << clone(param_type));

                // Add the parameter to the partial function.
                func_params << clone(param);

                // Add the argument to the create call inside the partial
                // function.
                create_args << (Expr << (RefLet << clone(param_id)));
              }

              auto create_func = Function
                << Implicit << Rhs << (Ident ^ l_create) << TypeParams
                << create_params << typevar(_) << DontCare << typepred()
                << (Block << (Expr << call(fq_new, new_args)));
              classbody << create_func;

              // Create the `new` function.
              classbody
                << (Function << Explicit << Rhs << (Ident ^ l_new) << TypeParams
                             << new_params << typevar(_) << DontCare
                             << typepred() << (Block << (Expr << unit())));

              // Create the partial function that returns the anonymous class.
              auto fq_create = append_fq(
                fq_class,
                selector(l_create, typeparams_to_typeargs(create_func)));

              ret << classdef
                  << (Function
                      << Implicit << hand << clone(id) << func_tp << func_params
                      << typevar(_) << DontCare << typepred()
                      << (Block
                          << (Expr << call(clone(fq_create), create_args))));

              fqs.push_back(fq_create);
              classbodies.push_back(classbody);
            }

            // We have an fq and a classdef for each arity. Now we need to
            // create the apply functions for each arity.
            for (auto arity = start_arity; arity < end_arity; ++arity)
            {
              // The anonymous class has a function for each intermediate arity
              // and for the final arity, allowing any number of arguments to be
              // supplied.
              auto classbody = classbodies[arity - start_arity];

              for (auto i = arity + 1; i <= end_arity; ++i)
              {
                // Build an apply function with (i - arity) parameters.
                auto self_id = Ident ^ _.fresh(l_self);
                Node apply_tp = TypeParams;

                // TODO: capability for Self, depends on captured param types
                Node apply_params = Params
                  << (Param << self_id << (Type << Self));
                Node apply_pred;
                Node fwd_args = Tuple;

                for (size_t j = 0; j < arity; ++j)
                {
                  // Include our captured arguments.
                  fwd_args
                    << (Expr << call(
                          selector(params->at(j) / Ident),
                          RefLet << clone(self_id)));
                }

                for (auto j = arity; j < i; ++j)
                {
                  // Add the additional parameters passed to this apply
                  // function.
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
                  fwd = clone(fq_f);
                }
                else
                {
                  // Intermediate arities create a new anonymous class for the
                  // next arity.
                  apply_pred = typepred();
                  fwd = clone(fqs[i - start_arity]);
                }

                classbody
                  << (Function << Implicit << hand << (Ident ^ l_apply)
                               << apply_tp << apply_params << typevar(_)
                               << DontCare << apply_pred
                               << (Block << (Expr << call(fwd, fwd_args))));
              }
            }

            return ret;
          }),
      }};
  }
}
