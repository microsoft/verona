#include "../lang.h"

namespace vc
{
  // Sythetic locations.
  inline const auto l_lambda = Location("lambda");

  Node call_func(Node ident, Node tps, Node args)
  {
    auto cls = ident->parent(ClassDef);
    auto func = ident->parent(Function);
    auto cls_path = scope_path(cls);
    auto cls_ta = fq_typeargs(cls_path, cls / TypeParams);
    auto func_ta = fq_typeargs(scope_path(func), tps);

    Node fn = FuncName;

    for (auto& s : cls_path)
    {
      if (s == cls)
        fn << (NameElement << clone(cls / Ident) << cls_ta);
      else
        fn << (NameElement << clone(s / Ident) << TypeArgs);
    }

    fn << (NameElement << clone(ident) << func_ta);
    return Call << fn << args;
  }

  AnonClass desugar_lambda(Match& _, Node lambda)
  {
    auto params = lambda / Params;
    auto type = lambda / Type;
    auto body = lambda / Body;

    for (auto& param : *params)
    {
      if (param == Expr)
        return {
          Node{}, err(param, "Can't use a case value in a non-case lambda")};
    }

    std::map<Location, std::pair<Node, bool>> freevars;
    std::set<Location> enclosing_field_captures;
    bool has_raise = false;
    bool has_var_capture = false;

    lambda->traverse([&](auto node) {
      bool ok = true;

      if (node == Error)
      {
        ok = false;
      }
      else if (node == Raise)
      {
        has_raise = true;
      }
      else if (node == LocalId)
      {
        if (node->lookup(lambda).empty())
        {
          auto loc = node->location();

          if (freevars.find(loc) == freevars.end())
          {
            // Look up the definition in the enclosing scope
            // (no boundary) to get its type.
            Node fv_type;
            auto defs = node->lookup();

            for (auto& def : defs)
            {
              if (def->in({ParamDef, Let, Var}))
              {
                bool fv_is_var = (def == Var);

                if (fv_is_var)
                  has_var_capture = true;

                fv_type = clone(def / Type);
                freevars.emplace(loc, std::make_pair(fv_type, fv_is_var));
                break;
              }
            }

            if (freevars.find(loc) == freevars.end())
            {
              // Check if this is a field of the enclosing lambda class.
              // This handles nested lambdas: the outer lambda captured
              // this variable as a field, and the inner lambda should
              // independently capture the field value.
              auto enclosing_cls = lambda->parent(ClassDef);

              if (enclosing_cls)
              {
                auto cls_body = enclosing_cls / ClassBody;

                for (auto& child : *cls_body)
                {
                  if (child == FieldDef && (child / Ident)->location() == loc)
                  {
                    fv_type = clone(child / Type);
                    freevars.emplace(loc, std::make_pair(fv_type, false));
                    enclosing_field_captures.insert(loc);
                    break;
                  }
                }
              }

              if (freevars.find(loc) == freevars.end())
                freevars.emplace(loc, std::make_pair(Node{}, false));
            }
          }
        }
      }

      return ok;
    });

    auto id = _.fresh(l_lambda);

    bool is_block = has_raise || has_var_capture;

    // Build scope info.
    auto enclosing_cls = lambda->parent(ClassDef);
    assert(enclosing_cls);
    auto cls_path = scope_path(enclosing_cls);

    // Annotate Raise nodes with the enclosing function's return
    // type, so the type checker can verify the raised value
    // against the function that will actually receive it.
    // This must happen before rewrite_typeparam_refs so the
    // type param references get rewritten correctly.
    if (has_raise)
    {
      auto enclosing_func = lambda->parent(Function);
      assert(enclosing_func);
      auto func_ret_type = enclosing_func / Type;

      lambda->traverse([&](auto node) {
        if (node != lambda && node == Lambda)
          return false;
        if (node == Raise && node->size() == 1)
          node << clone(func_ret_type);
        return node != Error;
      });
    }

    // Collect and rewrite free type params.
    auto free_tps = collect_free_typeparams(lambda);
    rewrite_typeparam_refs(lambda, free_tps, cls_path, id);

    // Build fields from free variables.
    std::vector<AnonClassField> fields;

    // For block lambdas (raising or var-capturing), add a
    // $raise_target field that captures the current raise target
    // at creation time.
    if (is_block)
    {
      auto u64_type = Type
        << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                     << (NameElement << (Ident ^ "u64") << TypeArgs));
      fields.push_back({Location("$raise_target"), u64_type, Expr << GetRaise});
    }

    for (auto& [freevar, fv_info] : freevars)
    {
      auto& [fv_type, fv_is_var] = fv_info;
      Node fv_resolved;

      if (fv_type)
      {
        fv_resolved = clone(fv_type);
        rewrite_typeparam_refs(fv_resolved, free_tps, cls_path, id);
      }
      else
      {
        fv_resolved = make_type();
      }

      if (fv_is_var)
      {
        // Capture a RegisterRef to the var.
        // Field type is ref[T] since the field holds a reference.
        auto ref_type = Type
          << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                       << (NameElement << (Ident ^ "ref")
                                       << (TypeArgs << clone(fv_resolved))));
        fields.push_back(
          {freevar,
           ref_type,
           Expr << (Ref << (Expr << (LocalId ^ freevar))),
           true});
      }
      else if (enclosing_field_captures.count(freevar) > 0)
      {
        // Enclosing lambda field capture: the creation-site expression
        // loads the field from the outer lambda's $self. At the creation
        // site, $self is the outer lambda's closure object.
        fields.push_back(
          {freevar,
           fv_resolved,
           Expr
             << (Load
                 << (Expr
                     << (FieldRef << (Expr << (LocalId ^ "$self"))
                                  << (FieldId ^ freevar))))});
      }
      else
      {
        fields.push_back({freevar, fv_resolved, Expr << (LocalId ^ freevar)});
      }
    }

    // Build apply body: field-loading preamble + lambda body.
    Node apply_body = Body;

    // Collect names needing body rewriting.
    // var captures: rewritten through Load/Ref via $ref_ locals.
    // let captures: rewritten to FieldRef on $self.
    std::set<Location> ref_captures;
    std::set<Location> let_captures;

    for (auto& field : fields)
    {
      if (field.is_var)
      {
        // Load the RegisterRef from the field into a $ref_
        // prefixed Let. The body will be rewritten to
        // read/write through this ref.
        auto ref_name = Location("$ref_" + std::string(field.name.view()));
        apply_body
          << (Expr
              << (Equals
                  << (Expr << (Let << (Ident ^ ref_name) << clone(field.type)))
                  << (Expr
                      << (Load
                          << (Expr
                              << (FieldRef << (Expr << (LocalId ^ "$self"))
                                           << (FieldId ^ field.name)))))));
        ref_captures.insert(field.name);
      }
      else if (field.name.view()[0] == '$')
      {
        // Internal field (e.g., $raise_target): load value
        // directly into a Let with the same name.
        apply_body
          << (Expr
              << (Equals << (Expr
                             << (Let << (Ident ^ field.name)
                                     << clone(field.type)))
                         << (Expr
                             << (Load
                                 << (Expr
                                     << (FieldRef
                                         << (Expr << (LocalId ^ "$self"))
                                         << (FieldId ^ field.name)))))));
      }
      else
      {
        // User let-capture: no preamble needed.
        // The body will be rewritten to access self.field
        // directly via FieldRef.
        let_captures.insert(field.name);
      }
    }

    // For block lambdas, set the raise target from the
    // captured field before executing the lambda body.
    if (is_block)
    {
      apply_body
        << (Expr << (SetRaise << (Expr << (LocalId ^ "$raise_target"))));
    }

    apply_body << *body;

    // Rewrite let-captured references in the body.
    // Reads become Load << (Expr << FieldRef($self, field)).
    // Writes (bare LHS of Equals) become FieldRef($self, field).
    // ANF handles read/write semantics based on position.
    if (!let_captures.empty())
    {
      std::vector<Node> to_rewrite;
      apply_body->traverse([&](auto node) {
        if (node == Lambda)
          return false;
        if (node == LocalId && let_captures.count(node->location()) > 0)
          to_rewrite.push_back(node);
        return true;
      });

      for (auto& node : to_rewrite)
      {
        auto parent = node->parent();

        // Check if this is an l-value (bare LHS of Equals).
        bool is_lvalue = false;

        if (parent && parent == Expr && parent->size() == 1)
        {
          auto grandparent = parent->parent();

          if (
            grandparent && grandparent == Equals &&
            grandparent->front() == parent)
          {
            is_lvalue = true;
          }
        }

        if (is_lvalue)
        {
          // Write: replace with Ref(Expr(FieldRef)) — ANF handles Store.
          parent->replace(
            node,
            Ref
              << (Expr
                  << (FieldRef << (Expr << (LocalId ^ "$self"))
                               << (FieldId ^ node->location()))));
        }
        else
        {
          // Read: replace with Load(FieldRef).
          parent->replace(
            node,
            Load
              << (Expr
                  << (FieldRef << (Expr << (LocalId ^ "$self"))
                               << (FieldId ^ node->location()))));
        }
      }
    }

    // Rewrite var-captured (ref) references in the body.
    // Reads become Load << (Expr << (LocalId ^ "$ref_x")).
    // Writes (bare LHS of Equals) become Ref << (LocalId ^ "$ref_x").
    if (!ref_captures.empty())
    {
      std::vector<Node> to_rewrite;
      apply_body->traverse([&](auto node) {
        if (node == LocalId && ref_captures.count(node->location()) > 0)
          to_rewrite.push_back(node);
        return true;
      });

      for (auto& node : to_rewrite)
      {
        auto ref_name =
          Location("$ref_" + std::string(node->location().view()));

        // Check if this is an l-value (LHS of Equals).
        // Only bare assignments (LocalId is the sole child of the
        // LHS Expr) are l-values. Compound expressions like
        // var(i) = val are reads (the indexing produces the ref).
        auto parent = node->parent();
        bool is_lvalue = false;

        if (parent && parent == Expr && parent->size() == 1)
        {
          auto grandparent = parent->parent();

          if (
            grandparent && grandparent == Equals &&
            grandparent->front() == parent)
          {
            is_lvalue = true;
          }
        }

        if (is_lvalue)
        {
          // Write: replace LocalId with
          // (Ref << (Expr << (LocalId ^ "$ref_x")))
          parent->replace(node, Ref << (Expr << (LocalId ^ ref_name)));
        }
        else
        {
          // Read: replace LocalId with
          // (Load << (Expr << (LocalId ^ "$ref_x")))
          parent->replace(node, Load << (Expr << (LocalId ^ ref_name)));
        }
      }
    }

    return make_anon_class(
      id, lambda, free_tps, fields, params, type, apply_body, is_block);
  }

  Node desugar_case_lambda(Match& _, Node lambda)
  {
    Node params = lambda / Params;
    Node type = lambda / Type;
    auto arity = params->size();

    if (!arity)
      return err(lambda, "Case must have at least one parameter");

    Node new_params = Params
      << (ParamDef << (Ident ^ "$arg") << type_any() << Body);
    Node new_type;
    Node new_body = Body;

    if (type->front() != TypeVar)
      new_type = Type << (Union << clone(type) << type_nomatch());
    else
      new_type = make_type();

    // We'll use this multiple times.
    auto nomatch_block = Block
      << (Body
          << (Return
              << (Expr
                  << (Call
                      << (FuncName
                          << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                          << (NameElement << (Ident ^ "nomatch") << TypeArgs)
                          << (NameElement << (Ident ^ "create") << TypeArgs))
                      << Args))));

    // Create the match values at the call site.
    Node callsite = ExprSeq;

    // Check arity of $arg.
    new_body
      << (Expr
          << (If << (Expr << (Unop << Len
                                   << (Args << (Expr << (LocalId ^ "$arg"))))
                          << (MethodName << (SymbolId ^ "!=") << TypeArgs)
                          << (Int ^ std::to_string(arity)))
                 << clone(nomatch_block)));

    // Destructure $arg into the original params.
    Node dst = Tuple;
    Node type_checks = Body;
    Node bindings = Body;

    for (auto& param : *params)
    {
      if (param == ParamDef)
      {
        // Destructure into a temporary so the arm binder is only introduced
        // after the type test succeeds.
        auto id = param / Ident;
        auto value_name = _.fresh(l_local);
        dst << (Expr << (Let << (Ident ^ value_name) << make_type()));

        // Test against the param type.
        type_checks
          << (Expr
              << (If << (Expr
                         << (Unop
                             << Not
                             << (Args
                                 << (Expr
                                     << (Typetest
                                         << (Expr << (LocalId ^ value_name))
                                         << clone(param / Type))))))
                     << clone(nomatch_block)));

        bindings
          << (TypedMove << clone(id) << (LocalId ^ value_name)
                        << clone(param / Type));
      }
      else
      {
        // Destructure the param out of $arg.
        auto dest_name = _.fresh(l_local);
        dst << (Expr << (Let << (Ident ^ dest_name) << make_type()));

        // Create the match value at the call site.
        // Type as `any` since the match comparison is dynamic dispatch.
        auto case_name = _.fresh(l_local);
        callsite
          << (Expr
              << (Equals << (Expr << (Let << (Ident ^ case_name) << type_any()))
                         << clone(param)));

        // Try calling == on the destructured value with the case value.
        // If == doesn't exist or returns non-bool, we get nomatch.
        auto eq_name = _.fresh(l_local);
        type_checks
          << (Expr
              << (Equals << (Expr << (Let << (Ident ^ eq_name) << make_type()))
                         << (Expr
                             << (TryCallDyn
                                 << (Expr << (LocalId ^ dest_name))
                                 << (SymbolId ^ "==") << TypeArgs
                                 << (Args
                                     << (Expr << (LocalId ^ case_name)))))));

        // If the result is not a bool, return nomatch.
        auto bool_type = Type
          << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                       << (NameElement << (Ident ^ "bool") << TypeArgs));
        type_checks
          << (Expr
              << (If << (Expr
                         << (Unop << Not
                                  << (Args
                                      << (Expr
                                          << (Typetest
                                              << (Expr << (LocalId ^ eq_name))
                                              << bool_type)))))
                     << clone(nomatch_block)));

        // If the bool is false (values not equal), return nomatch.
        type_checks
          << (Expr
              << (If << (Expr
                         << (Unop << Not
                                  << (Args << (Expr << (LocalId ^ eq_name)))))
                     << clone(nomatch_block)));
      }
    }

    // For a single param, assign directly without Tuple wrapping.
    if (dst->size() == 1)
      dst = dst->front();
    else
      dst = Expr << dst;

    // Emit the destructuring assignment.
    new_body << (Expr << (Equals << dst << (Expr << (LocalId ^ "$arg"))));

    // Emit type checks after destructuring.
    new_body << *type_checks;

    // Bind typed pattern names only on the successful path.
    new_body << *bindings;

    // Proceed with the original body.
    new_body << *(lambda / Body);

    // Create the new lambda. This will be fully desugared later.
    auto new_lambda = Lambda << new_params << new_type << new_body;
    wfPassSugar.build_st(new_lambda);
    return callsite << (Expr << new_lambda);
  }

  PassDef sugar()
  {
    PassDef p{
      "sugar",
      wfPassSugar,
      dir::topdown,
      {
        // Rewrite `when` arguments.
        In(Expr) * T(When)
            << ((T(Expr)[Args] << (T(Tuple)[Tuple] * End)) * T(Type)[Type] *
                (T(Expr)[Expr] << T(Lambda)[Lambda])) >>
          [](Match& _) {
            if (_(Tuple)->size() != (_(Lambda) / Params)->size())
            {
              return err(_(Tuple), "When argument count must match lambda")
                << errmsg("Lambda is here:") << errloc(_(Lambda));
            }

            return When << (Args << *_(Tuple)) << _(Type) << _(Expr);
          },

        In(Expr) * T(When)
            << (T(Expr)[Lhs] * T(Type)[Type] *
                (T(Expr)[Expr] << T(Lambda)[Lambda])) >>
          [](Match& _) {
            if ((_(Lambda) / Params)->size() != 1)
            {
              return err(_(Lhs), "When argument count must match lambda")
                << errmsg("Lambda is here:") << errloc(_(Lambda));
            }

            return When << (Args << _(Lhs)) << _(Type) << _(Expr);
          },

        // Match.
        In(Expr) * T(MatchExpr) << (T(Expr)[Expr] * T(ExprSeq)[ExprSeq]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            Node seq = Seq;
            Node callsite = ExprSeq;

            callsite
              << (Expr
                  << (Equals << (Expr << (Let << (Ident ^ id) << make_type()))
                             << _(Expr)));

            auto arms = _(ExprSeq);
            Node last;

            for (auto& arm : *arms)
            {
              auto lambda = arm->front();
              assert(lambda == Lambda);
              auto call_case =
                (Expr << desugar_case_lambda(_, lambda) << (LocalId ^ id));

              if (!last)
              {
                // First arm is just an Expr.
                last = call_case;
              }
              else
              {
                // Build the else chain.
                last = Expr << (Else << last << (Block << (Body << call_case)));
              }
            }

            return seq << (callsite << last);
          },

        // Turn lambdas into anonymous classes.
        In(Expr) * T(Lambda)[Lambda] >>
          [](Match& _) {
            auto result = desugar_lambda(_, _(Lambda));
            return Seq << (Lift << ClassBody << result.class_def)
                       << result.create_call;
          },

        // Default arguments.
        In(ClassBody) * T(Function)[Function]
            << (T(Lhs, Rhs)[Lhs] * T(Ident, SymbolId)[Ident] *
                T(TypeParams)[TypeParams] * T(Params)[Params] * T(Type)[Type] *
                T(Where)[Where] * T(Body)[Body]) >>
          [](Match& _) -> Node {
          auto params = _(Params);
          if (params->empty())
            return NoChange;

          auto last_param = params->back();
          auto body = last_param / Body;

          if (body->empty())
            return NoChange;

          // Generate a function with one fewer parameters.
          auto ident = _(Ident);
          auto params_0 = clone(params);
          params_0->pop_back();

          // Remove the default value from all parameters.
          for (auto& param : *params)
            (param / Body) = Body;

          Node args = Args;

          // Call the original function with the default final argument.
          for (auto& param : *params_0)
            args << (Expr << (LocalId ^ (param / Ident)));

          args << (Expr << (ExprSeq << *body));

          return Seq << (Function << clone(_(Lhs)) << clone(ident)
                                  << clone(_(TypeParams)) << params_0
                                  << clone(_(Type)) << clone(_(Where))
                                  << (Body
                                      << (Expr << call_func(
                                            ident, _(TypeParams), args))))
                     << _(Function);
        },

        // Auto-RHS.
        In(ClassBody) * T(Function)[Function]
            << (T(Lhs) * T(Ident, SymbolId)[Ident] * T(TypeParams)[TypeParams] *
                T(Params)[Params] * T(Type)[Type] * T(Where)[Where]) >>
          [](Match& _) -> Node {
          // Check if an RHS function with the same name exists that covers
          // this arity. An RHS function with default arguments will expand
          // to cover arities from its minimum arity (without defaults) to
          // its full arity. Don't generate an auto-RHS if a default argument
          // expansion will cover this arity.
          auto ident = _(Ident);
          auto arity = _(Params)->size();
          auto cls = _(Function)->parent(ClassBody);

          for (auto& def : *cls)
          {
            if (
              (def == Function) && ((def / Lhs) == Rhs) &&
              (def / Ident)->equals(ident))
            {
              auto rhs_params = def / Params;
              auto rhs_arity = rhs_params->size();

              // Compute minimum arity by excluding trailing default args.
              auto rhs_min_arity = rhs_arity;

              while ((rhs_min_arity > 0) &&
                     !(rhs_params->at(rhs_min_arity - 1) / Body)->empty())
                rhs_min_arity--;

              if ((arity >= rhs_min_arity) && (arity <= rhs_arity))
                return NoChange;
            }
          }

          // Forward the arguments.
          Node args = Args;

          for (auto& param : *_(Params))
            args << (Expr << (LocalId ^ (param / Ident)));

          // Create the RHS function.
          auto rhs =
            Function << Rhs << clone(ident) << clone(_(TypeParams))
                     << clone(_(Params)) << make_type() << clone(_(Where))
                     << (Body
                         << (Expr
                             << (Load
                                 << (Expr << Ref
                                          << call_func(
                                               ident, _(TypeParams), args)))));

          return Seq << _(Function) << rhs;
        },
      }};

    p.post([](auto top) {
      top->traverse([&](auto node) {
        bool ok = true;

        if (node == Error)
        {
          ok = false;
        }
        else if (node->get_contains_error())
        {
          // Do nothing.
        }
        else if (node == ParamDef)
        {
          auto body = node / Body;

          if (body->empty())
          {
            node->replace(body);
          }
          else
          {
            node->replace(
              body, err(body, "Default arguments must be at the end"));
            ok = false;
          }
        }

        return ok;
      });

      return 0;
    });

    return p;
  }
}
