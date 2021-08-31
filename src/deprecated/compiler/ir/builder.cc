// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/builder.h"

#include "compiler/assignssymbols.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  struct ScopeData
  {
    std::vector<Variable> temporaries;
    std::vector<LocalID> locals;

    ScopeData() = default;

    ScopeData(const ScopeData&) = delete;
    ScopeData& operator=(const ScopeData&) = delete;
  };

  /**
   * Lowering an expression to IR depends on how the value is used.
   * For example, in `x = y`, we want `y` to be copied, where as in `x = y.f`,
   * it should not be.
   *
   * The ValueKind variant is passed to the visitor and controls how lowering is
   * performed.
   */
  struct ValueKindAny;
  struct ValueKindOwned;
  struct ValueKindUnused;
  typedef std::variant<ValueKindAny, ValueKindOwned, ValueKindUnused> ValueKind;

  /**
   * The caller of visit_expr does not own the resulting variable. If the
   * expression is a local variable, it will be returned. Otherwise a temporary
   * is allocated and owned by the specified scope.
   */
  struct ValueKindAny
  {
    ScopeData* scope;
    explicit ValueKindAny(ScopeData& scope) : scope(&scope) {}
  };

  /**
   * The caller of visit_expr owns the resulting variable. It is responsible for
   * killing it.
   */
  struct ValueKindOwned
  {};

  /**
   * The expression's result is unused. If the expression has no-side effect,
   * lowering could lead to no code being generated.
   */
  struct ValueKindUnused
  {};

  /**
   * Result of lowering an expression to IR.
   *
   * It may hold a BuilderResult::Invalid, only if ValueKind::Unused was used
   * when calling the visit_expr method. In this case the caller must ignore the
   * value.
   *
   * Otherwise, if ValueKind::Unused was not passed, the BuilderResult must
   * contain a valid Variable, and may be implicitly converted to that Variable.
   *
   * Attempting to convert an invalid BuilderResult to a Variable will cause a
   * runtime abort.
   *
   * This is generic over its contents, so it can be used with either a Variable
   * or an IRInput.
   */
  template<typename T>
  struct BuilderResult
  {
    /**
     * Token value used in place of a Variable, when the result is unused and no
     * code was generated.
     */
    struct Invalid
    {};

    BuilderResult(const T& inner) : inner(inner) {}
    BuilderResult(const Invalid& inner) : inner(inner) {}

    /**
     * Enable conversion from BuilderResult<U> to BuilderResult<T>, if U can be
     * converted to T.
     *
     * This is typically used to convert a BuilderResult<IRInput> to a
     * BuilderResult<Variable>.
     */
    template<typename U>
    BuilderResult(const BuilderResult<U>& other)
    {
      if (other.is_valid())
        inner = *other;
      else
        inner = Invalid();
    }

    bool is_valid() const
    {
      return std::holds_alternative<T>(inner);
    }

    const T& operator*() const
    {
      if (std::holds_alternative<T>(inner))
        return std::get<T>(inner);
      else
        throw std::logic_error("Invalid BuilderResult not ignored.");
    }

    std::variant<T, Invalid> inner;
  };

  /**
   * Augment a BuilderResult<Variable> into a BuilderResult<IRInput> by adding a
   * source range to it.
   */
  static BuilderResult<IRInput> wrap_variable(
    BuilderResult<Variable> result, SourceManager::SourceRange source_range)
  {
    if (result.is_valid())
      return IRInput(*result, source_range);
    else
      return BuilderResult<IRInput>::Invalid();
  }

  /**
   * Unwrap a BuilderResult<Variable> into a Variable. This method is overloaded
   * to also work with a Variable argument, for use in places that are generic
   * over either.
   */
  static Variable unwrap_variable(BuilderResult<Variable> v)
  {
    return *v;
  }
  static Variable unwrap_variable(Variable v)
  {
    return v;
  }

  class IRBuilder
  : private ExprVisitor<BuilderResult<IRInput>, ValueKind, BasicBlock*&>
  {
    friend std::unique_ptr<MethodIR>
    build_ir(const FnSignature& sig, const FnBody& body);

  private:
    IRBuilder(MethodIR* ir) : method_ir_(ir)
    {
      scopes_.push_back(std::make_unique<ScopeData>());
    }

    /**
     * Queue building a when expression.  Returns the index where the closure
     * will be added into once complete.
     */
    size_t queue_build(const WhenExpr& when)
    {
      std::vector<LocalID> params;

      for (const auto& cown : when.cowns)
      {
        params.push_back(cown->get_binder());
      }
      for (auto& capture : when.captures)
      {
        params.push_back(capture);
      }

      return queue_build_body(std::nullopt, params, &*when.body);
    }

    void build(
      std::optional<LocalID> receiver,
      std::vector<LocalID> params,
      Expression* body)
    {
      auto ir = method_ir_->create_function_ir();
      function_ir_ = ir;
      BasicBlock* bb = ir->add_block(/* dominator */ nullptr);
      ir->entry = bb;

      /**
       * The builder is already initialized with a "cown" scope.
       * This scope only owns the return value. All other variables, including
       * function arguments, belong to inner scopes.
       *
       * The cown scope does not get any EndScopeStmt, since we want the return
       * value to still be usable by the ReturnTerminator.
       *
       * The "body" scope is the owner of the function arguments and is used to
       * generate the function body. It ends just before the final return
       * terminator.
       */
      ScopeData& cown_scope = current_scope();

      IRInput output = *push_scope(
        [&](ScopeData& body_scope) {
          auto add_var = [this, &body_scope, &ir, &bb](auto expr) {
            // Variable is owned by the local binding
            Variable var = this->fresh_variable(ValueKindOwned(), expr);
            ir->variable_definitions.insert({var, ir->entry->entry_point()});

            add_definition(bb, expr, var);
            body_scope.locals.push_back(expr);
            return var;
          };

          if (receiver.has_value())
          {
            ir->receiver = add_var(receiver.value());
          }
          for (const auto& param : params)
          {
            ir->parameters.push_back(add_var(param));
          }
        },
        *body,
        ValueKindAny(cown_scope),
        bb);

      set_terminator(bb, ReturnTerminator{output});
      function_ir_->exits.insert(bb);

      for (const auto& bb : ir->basic_blocks)
      {
        assert(!has_pending(&bb));
      }

      if (!worklist.empty())
      {
        auto [recv, params, body] = worklist.front();
        worklist.pop();
        build(recv, params, body);
      }
    }

    /**
     * Allocate a new SSA variable.
     */
    Variable fresh_variable(
      const ValueKind kind, std::optional<LocalID> lid = std::nullopt)
    {
      Variable v = {next_variable_++, lid};

      match(
        kind,
        [&](const ValueKindAny& eph) {
          // Make the variable a temporary in the scope that requested it.
          eph.scope->temporaries.push_back(v);
        },
        [&](const ValueKindOwned&) {
          // The caller is responsible for giving this variable an owner.
        },
        [&](const ValueKindUnused&) {
          current_scope().temporaries.push_back(v);
        });

      return v;
    }

    TypeArgumentsId fresh_type_arguments()
    {
      return TypeArgumentsId(next_type_arguments_++);
    }

    /**
     * Add (or replace) a mapping from local name to an SSA variable.
     */
    void add_definition(BasicBlock* bb, LocalID local, Variable v)
    {
      definitions_[{bb, local}] = v;
    }

    Variable find_variable(LocalID local, BasicBlock* bb)
    {
      if (auto it = definitions_.find({bb, local}); it != definitions_.end())
      {
        return it->second;
      }

      if (bb->predecessors.empty())
      {
        InternalError::print("cannot find symbol {}\n", local);
      }

      if (bb->predecessors.size() == 1 && !has_pending(bb))
      {
        auto output = find_variable(local, bb->predecessors[0]);
        add_definition(bb, local, output);
        return output;
      }

      // Check if this is an expression we want to handle specially
      // i.e. a loop.
      auto expr_bb = original_expr_.find(bb);
      if (expr_bb != original_expr_.end())
      {
        if (!assigns_sym_.assigns(((*expr_bb).second.first), local))
        {
          auto output = find_variable(local, (*expr_bb).second.second);
          add_definition(bb, local, output);
          return output;
        }
        else
        {
          if (has_pending(bb))
          {
            // The list of predecessors is not fully known yet.
            // We use a fresh variable, the phi node will be defined later when
            // finish_block() the BB.
            //
            // Variable is owned by the local binding
            Variable output = fresh_variable(ValueKindOwned());
            pending_phis_[bb].push_back({local, output});
            add_definition(bb, local, output);
            return output;
          }
        }
      }

      std::vector<std::pair<BasicBlock*, Variable>> inputs;
      for (BasicBlock* predecessor : bb->predecessors)
      {
        Variable var = find_variable(local, predecessor);
        inputs.push_back({predecessor, var});
      }

      // Insert a definition using a dummy Variable. If this succeeds, we create
      // a Phi node to replace the dummy value.
      //
      // Insertion could fail if we're inside a loop. Infinite recursion was
      // prevented by creating an empty Phi node. Now that we know all the
      // inputs, we may update the Phi node with them.
      auto it = definitions_.insert({std::make_pair(bb, local), Variable{0}});
      if (it.second)
      {
        // Variable is owned by the local binding
        it.first->second = *build_phi_node(bb, inputs, ValueKindOwned());
      }
      else
      {
        finish_phi_node(bb, inputs, it.first->second);
      }

      return it.first->second;
    }

    /**
     * Add a statement of type T to the given basic block.
     *
     * T must have a `Variable output` field. This method will add that Variable
     * to the BasicBlock's `variable_definitions` map with the appropriate
     * IRPoint.
     *
     * That output variable along with the source range of the statement are
     * returned as an IRInput.
     */
    template<typename T>
    IRInput add_statement(BasicBlock* bb, T stmt)
    {
      size_t index = bb->statements.size();
      IRPoint point = IRPoint::statement(bb, index);

      Variable output = stmt.output;
      SourceManager::SourceRange source_range = stmt.source_range;
      bb->statements.push_back(std::move(stmt));

      function_ir_->variable_definitions.insert({output, point});

      return IRInput(output, source_range);
    }

    /**
     * Add a statement of type T to the given basic block.
     *
     * This method must only be called with statement types that don't have an
     * output.
     */
    template<typename T>
    void add_outputless_statement(BasicBlock* bb, T stmt)
    {
      // Make sure we don't accidentally use this method with statements that do
      // have an output. These are currently the only statements that don't.
      static_assert(
        std::is_same_v<T, OverwriteStmt> || std::is_same_v<T, EndScopeStmt>);
      bb->statements.push_back(std::move(stmt));
    }

    /**
     * Set the terminator of the given basic block.
     *
     * This method will add to basic block as a predecessor in any basic block
     * the terminator points to.
     */
    template<typename T>
    const T& set_terminator(BasicBlock* bb, T term)
    {
      assert(!bb->terminator.has_value());

      add_terminator_predecessor(bb, term);
      bb->terminator = term;
      return std::get<T>(*bb->terminator);
    }

    bool has_pending(const BasicBlock* bb)
    {
      return pending_blocks_.find(bb) != pending_blocks_.end();
    }

    void set_pending(const BasicBlock* bb)
    {
      pending_blocks_.insert(bb);
    }

    void finish_block(BasicBlock* bb)
    {
      assert(has_pending(bb));

      pending_blocks_.erase(bb);

      auto phis_it = pending_phis_.find(bb);
      if (phis_it == pending_phis_.end())
        return;

      for (auto [symbol, output] : phis_it->second)
      {
        std::vector<std::pair<BasicBlock*, Variable>> inputs;
        for (BasicBlock* predecessor : bb->predecessors)
        {
          Variable var = find_variable(symbol, predecessor);
          inputs.push_back({predecessor, var});
        }
        finish_phi_node(bb, inputs, output);
      }
    }

    /**
     * Create a new Phi node at the top of `bb`.
     *
     * `V` may be either Variable or BuilderResult<Variable>.
     *
     * If `V` is BuilderResult<Variable> and kind != Unused, all the inputs must
     * be valid.
     */
    template<typename V>
    BuilderResult<Variable> build_phi_node(
      BasicBlock* bb,
      const std::vector<std::pair<BasicBlock*, V>>& inputs,
      ValueKind kind)
    {
      static_assert(
        std::is_same_v<V, Variable> ||
          std::is_same_v<V, BuilderResult<Variable>>,
        "build_phi_node must be instantiated with Variable or "
        "BuilderResult<Variable>");

      assert(!inputs.empty());

      if (std::holds_alternative<ValueKindUnused>(kind))
        return BuilderResult<Variable>::Invalid();

      // From this point on, kind != Unused and all the inputs are valid.
      // Hence we can use unwrap_variable safely.
      Variable variable = unwrap_variable(inputs[0].second);
      auto is_same = [&](const auto& input) {
        return unwrap_variable(input.second) == variable;
      };

      if (std::all_of(inputs.begin() + 1, inputs.end(), is_same))
      {
        // The input should have been owned by the Phi node, but we're skipping
        // phi node creation. If kind is ValueKindOwned, that's okay because the
        // caller of build_phi_node() will just own that value instead.
        //
        // However if the kind is ValueKindAny, the caller does not take
        // ownership of the result. We therefore need to make it owned by the
        // specified scope instead.
        if (auto kind_any = std::get_if<ValueKindAny>(&kind))
        {
          kind_any->scope->temporaries.push_back(variable);
        }
        return variable;
      }

      Variable output = fresh_variable(kind);
      finish_phi_node(bb, inputs, output);

      return output;
    }

    /**
     * Update a pending Phi node with the given inputs.
     *
     * `V` may be either Variable or BuilderResult<Variable>.
     *
     * All the inputs must be valid, even if `V` is BuilderResult<Variable>.
     */
    template<typename V>
    void finish_phi_node(
      BasicBlock* bb,
      const std::vector<std::pair<BasicBlock*, V>>& inputs,
      Variable output)
    {
      static_assert(
        std::is_same_v<V, Variable> ||
          std::is_same_v<V, BuilderResult<Variable>>,
        "finish_phi_node must be instantiated with Variable or "
        "BuilderResult<Variable>");

      assert(!inputs.empty());

      for (auto [predecessor, variable] : inputs)
      {
        // `finish_phi_node` requires all inputs to be valid, even if `V` is
        // BuilderResult<Variable>.
        if constexpr (std::is_same_v<V, BuilderResult<Variable>>)
          assert(variable.is_valid());

        BranchTerminator& terminator = predecessor->branch_terminator();
        assert(bb->phi_nodes.size() == terminator.phi_arguments.size());
        terminator.phi_arguments.push_back(unwrap_variable(variable));
      }
      bb->phi_nodes.push_back(output);

      function_ir_->variable_definitions.insert({output, bb->entry_point()});
    }

    /**
     * Create a new scope and visit the given expression inside this scope.
     *
     * If a `setup` closure is given, it will be executed after the new scope is
     * pushed, but before the expression is visited. This can be used to
     * introduce new variables in this scope, before visiting the body (e.g. for
     * match arms).
     */
    BuilderResult<IRInput> push_scope(
      const std::function<void(ScopeData&)>& setup,
      Expression& expr,
      ValueKind kind,
      BasicBlock*& bb)
    {
      scopes_.push_back(std::make_unique<ScopeData>());

      ScopeData& scope_data = *scopes_.back();

      setup(scope_data);

      BuilderResult<IRInput> result = visit_expr(expr, kind, bb);

      if (!scope_data.temporaries.empty() || !scope_data.locals.empty())
      {
        EndScopeStmt stmt(expr.source_range);
        stmt.dead_variables = scope_data.temporaries;
        for (LocalID local : scope_data.locals)
        {
          stmt.dead_variables.push_back(find_variable(local, bb));
        }
        add_outputless_statement(bb, std::move(stmt));
      }

      scopes_.pop_back();

      return result;
    }

    BuilderResult<IRInput>
    push_scope(Expression& expr, ValueKind kind, BasicBlock*& bb)
    {
      return push_scope([](ScopeData&) {}, expr, kind, bb);
    }

    IRInput visit_input(Expression& expr, BasicBlock*& bb)
    {
      return *visit_expr(expr, ValueKindAny(current_scope()), bb);
    }

    BuilderResult<IRInput>
    visit_symbol(SymbolExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      return match(
        expr.symbol,
        [&](const LocalDef* local) {
          IRInput input(find_variable(local, bb), expr.source_range);

          /*
           * The `kind->scope == current_scope()` condition ensures that we
           * don't return a variable that would be going out of scope. For
           *
           *   eg. y.f = { var x = ...; x };
           *
           *   When evaluating x but with the outer scope we want to generate a
           *   copy.
           *
           *   This is stricter than what we really need, i.e.
           *   `local->scope_depth <= kind_any->scope_depth`
           */
          if (auto kind_any = std::get_if<ValueKindAny>(&kind);
              kind_any && kind_any->scope == &current_scope())
          {
            return input;
          }

          // We always make a copy in the Owned and Unused case. The latter
          // ensures we preserve any typesystem side-effects cause by an
          // extended liveness.
          CopyStmt stmt(expr.source_range);
          stmt.input = input;
          stmt.output = fresh_variable(kind);
          return add_statement(bb, std::move(stmt));
        },
        [&](const Entity* definition) {
          StaticTypeStmt stmt(expr.source_range);
          stmt.definition = definition;
          stmt.type_arguments = fresh_type_arguments();
          stmt.output = fresh_variable(kind);
          return add_statement(bb, std::move(stmt));
        },
        [&](const TypeParameterDef*) -> IRInput { abort(); },
        [&](ErrorSymbol) -> IRInput { abort(); });
    }

    BuilderResult<IRInput> visit_define_local(
      DefineLocalExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      current_scope().locals.push_back(expr.local);

      if (expr.right)
      {
        // Variable is owned by the local binding
        IRInput value = *visit_expr(*expr.right, ValueKindOwned(), bb);
        add_definition(bb, expr.local, value);
      }
      else
      {
        // Variable is owned by the local binding
        IRInput value = *unit(expr.source_range, ValueKindOwned(), bb);
        add_definition(bb, expr.local, value);
      }
      return unit(expr.source_range, kind, bb);
    }

    BuilderResult<IRInput> visit_assign_local(
      AssignLocalExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      // Variable is owned by the local binding
      IRInput new_value = *visit_expr(*expr.right, ValueKindOwned(), bb);

      Variable old_value = find_variable(expr.local, bb);

      OverwriteStmt stmt(expr.source_range);
      stmt.dead_variable = old_value;
      add_outputless_statement(bb, std::move(stmt));

      add_definition(bb, expr.local, new_value);

      return unit(expr.source_range, kind, bb);
    }

    BuilderResult<IRInput>
    visit_field(FieldExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      ReadFieldStmt stmt(expr.source_range);
      stmt.name = expr.field_name;
      stmt.base = visit_input(*expr.expr, bb);
      stmt.source_range = expr.source_range;
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput> visit_assign_field(
      AssignFieldExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      WriteFieldStmt stmt(expr.source_range);
      stmt.name = expr.field_name;
      stmt.base = visit_input(*expr.expr, bb);
      stmt.right = visit_input(*expr.right, bb);
      stmt.source_range = expr.source_range;
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput>
    visit_seq(SeqExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      for (const auto& e : expr.elements)
      {
        visit_expr(*e, ValueKindUnused(), bb);
      }

      return visit_expr(*expr.last, kind, bb);
    }

    BuilderResult<IRInput>
    visit_call(CallExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      CallStmt stmt(expr.source_range);
      stmt.receiver = visit_input(*expr.receiver, bb);
      stmt.method = expr.method_name;
      stmt.type_arguments = fresh_type_arguments();

      for (const auto& arg : expr.arguments)
      {
        stmt.arguments.push_back(visit_input(*arg->inner, bb));
      }

      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput>
    visit_match_expr(MatchExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      IRInput input = visit_input(*expr.expr, bb);

      std::vector<MatchTerminator::Arm> arms;
      for (const auto& arm : expr.arms)
      {
        BasicBlock* child_bb = function_ir_->add_block(bb);

        // Variable will be owned by the local binding
        Variable variable = fresh_variable(ValueKindOwned());
        arms.push_back(MatchTerminator::Arm{arm->type, child_bb, variable});
      }

      const MatchTerminator& terminator =
        set_terminator(bb, MatchTerminator{input, arms});

      // TODO: if there's only one arm, then that arm is actually a closer
      // dominator.
      BasicBlock* exit_block = function_ir_->add_block(bb);

      std::vector<std::pair<BasicBlock*, BuilderResult<Variable>>> phi_inputs;

      for (const auto& [ast_arm, ir_arm] : safe_zip(expr.arms, terminator.arms))
      {
        BasicBlock* child_bb = ir_arm.target;

        // Today in C++ is terrible:
        // https://stackoverflow.com/questions/46114214 ast_arm and ir_arm
        // aren't variables, meaning they aren't captured by
        // [&]. Hence the explicit binding. This will be fixed in C++20 (P1381).
        auto setup_bindings =
          [&, &ast_arm = ast_arm, &ir_arm = ir_arm](ScopeData& inner_scope) {
            LocalID local = ast_arm->local;
            Variable binding = ir_arm.binding;

            MatchBindStmt stmt(ast_arm->source_range);
            stmt.input = input;
            stmt.output = binding;
            stmt.type = ir_arm.type;
            add_statement(child_bb, std::move(stmt));

            inner_scope.locals.push_back(local);
            add_definition(child_bb, local, binding);
          };

        // Result of each branch is owned by the Phi node.
        BuilderResult result = push_scope(
          setup_bindings, *ast_arm.get()->expr, phi_input_kind(kind), child_bb);

        phi_inputs.push_back({child_bb, result});

        set_terminator(child_bb, BranchTerminator{exit_block});
      }

      bb = exit_block;

      return wrap_variable(
        build_phi_node(bb, phi_inputs, kind), expr.source_range);
    }

    BuilderResult<IRInput>
    visit_new_expr(NewExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      NewStmt stmt(expr.source_range);
      stmt.definition = expr.definition;
      stmt.type_arguments = fresh_type_arguments();

      if (expr.parent)
      {
        Variable parent = find_variable(expr.parent->local, bb);
        stmt.parent = IRInput(parent, expr.parent->source_range);
      }

      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput>
    visit_empty(EmptyExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      return unit(expr.source_range, kind, bb);
    }

    BuilderResult<IRInput> visit_integer_literal_expr(
      IntegerLiteralExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      IntegerLiteralStmt stmt(expr.source_range);
      stmt.value = expr.value.value;
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput> visit_string_literal_expr(
      StringLiteralExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      StringLiteralStmt stmt(expr.source_range);
      stmt.value = expr.value;
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput>
    visit_when(WhenExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      WhenStmt stmt(expr.source_range);

      for (const auto& cown : expr.cowns)
      {
        if (auto argument = dynamic_cast<WhenArgumentAs*>(cown.get()))
        {
          stmt.cowns.push_back(visit_input(*argument->inner, bb));
        }
        else if (auto argument = dynamic_cast<WhenArgumentShadow*>(cown.get()))
        {
          IRInput cown(
            find_variable(argument->local, bb), argument->source_range);
          stmt.cowns.push_back(cown);
        }
        else
        {
          abort();
        }
      }

      for (const auto& capture : expr.captures)
      {
        IRInput input(find_variable(capture, bb), expr.source_range);
        stmt.captures.push_back(input);
      }

      stmt.closure_index = queue_build(expr);

      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput>
    visit_while(WhileExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      BasicBlock* header_bb = function_ir_->add_block(bb);
      BasicBlock* body_bb = function_ir_->add_block(header_bb);
      BasicBlock* exit_bb = function_ir_->add_block(header_bb);

      original_expr_[header_bb] = {&expr, bb};
      // Precalc defined symbols to ensure top down traversal.
      assigns_sym_.precalc(&expr);

      set_pending(header_bb);

      set_terminator(bb, BranchTerminator{header_bb});

      IRInput condition = visit_input(*expr.condition, header_bb);

      set_terminator(header_bb, IfTerminator{condition, body_bb, exit_bb});

      push_scope(*expr.body, ValueKindUnused(), body_bb);

      set_terminator(body_bb, BranchTerminator{header_bb});
      finish_block(header_bb);

      bb = exit_bb;

      return unit(expr.source_range, kind, bb);
    }

    BuilderResult<IRInput>
    visit_if(IfExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      IRInput condition = visit_input(*expr.condition, bb);

      BasicBlock* then_bb = function_ir_->add_block(bb);
      BasicBlock* else_bb = function_ir_->add_block(bb);

      set_terminator(bb, IfTerminator{condition, then_bb, else_bb});

      // Result of each branch is owned by the Phi node.
      BuilderResult then_value =
        push_scope(*expr.then_block, phi_input_kind(kind), then_bb);

      BuilderResult<IRInput> else_value = BuilderResult<IRInput>::Invalid();

      if (expr.else_block)
      {
        else_value =
          push_scope(*expr.else_block->body, phi_input_kind(kind), else_bb);
      }
      else
      {
        else_value = unit(expr.source_range, phi_input_kind(kind), else_bb);
      }

      BasicBlock* exit_bb = function_ir_->add_block(bb);
      set_terminator(then_bb, BranchTerminator{exit_bb});
      set_terminator(else_bb, BranchTerminator{exit_bb});

      original_expr_[exit_bb] = {&expr, bb};
      // Precalc defined symbols to ensure top down traversal.
      assigns_sym_.precalc(&expr);

      bb = exit_bb;

      std::vector<std::pair<BasicBlock*, BuilderResult<Variable>>> phi_inputs =
        {{then_bb, then_value}, {else_bb, else_value}};

      return wrap_variable(
        build_phi_node(bb, phi_inputs, kind), expr.source_range);
    }

    BuilderResult<IRInput>
    visit_block(BlockExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      return push_scope(*expr.inner, kind, bb);
    }

    BuilderResult<IRInput>
    visit_view_expr(ViewExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      ViewStmt stmt(expr.source_range);
      stmt.input = visit_input(*expr.expr, bb);
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    BuilderResult<IRInput> visit_binary_operator_expr(
      BinaryOperatorExpr& expr, ValueKind kind, BasicBlock*& bb) final
    {
      IRInput left = visit_input(*expr.left, bb);
      IRInput right = visit_input(*expr.right, bb);

      CallStmt stmt(expr.source_range);
      stmt.receiver = left;
      stmt.method = binary_operator_method_name(expr.kind->value());
      stmt.type_arguments = fresh_type_arguments();
      stmt.arguments.push_back(right);
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    /**
     * Generate a unit value, if needed.
     *
     * If kind is Unused, BuilderResult::Invalid is returned. Otherwise a fresh
     * variable is created and a UnitStmt is generated to write to it.
     */
    BuilderResult<IRInput> unit(
      SourceManager::SourceRange source_range, ValueKind kind, BasicBlock* bb)
    {
      if (std::holds_alternative<ValueKindUnused>(kind))
        return BuilderResult<IRInput>::Invalid();

      UnitStmt stmt(source_range);
      stmt.output = fresh_variable(kind);
      return add_statement(bb, std::move(stmt));
    }

    /**
     * For each basic block `term` jumps to, add `pred` as a predecessor.
     */
    void
    add_terminator_predecessor(BasicBlock* pred, const ReturnTerminator& term)
    {}
    void
    add_terminator_predecessor(BasicBlock* pred, const BranchTerminator& term)
    {
      term.target->predecessors.push_back(pred);
    }
    void
    add_terminator_predecessor(BasicBlock* pred, const MatchTerminator& term)
    {
      for (const auto& arm : term.arms)
      {
        arm.target->predecessors.push_back(pred);
      }
    }
    void add_terminator_predecessor(BasicBlock* pred, const IfTerminator& term)
    {
      term.true_target->predecessors.push_back(pred);
      term.false_target->predecessors.push_back(pred);
    }

    /**
     * For a phi node with the given kind, get the kind required by its inputs.
     *
     * This will be ValueKindOwned, unless the Phi node is unused in which case
     * the inputs are also unused.
     */
    ValueKind phi_input_kind(ValueKind kind)
    {
      return match(
        kind,
        [&](const ValueKindAny&) -> ValueKind { return ValueKindOwned(); },
        [&](const ValueKindOwned&) -> ValueKind { return ValueKindOwned(); },
        [&](const ValueKindUnused&) -> ValueKind { return ValueKindUnused(); });
    }

    uint64_t next_variable_ = 0;
    uint64_t next_type_arguments_ = 0;

    /* Current function we are processing */
    FunctionIR* function_ir_;

    /* The set of irs associated with the method we are processing. */
    MethodIR* method_ir_;

    /*
      The list of irs to still be built.
    */
    std::queue<
      std::tuple<std::optional<LocalID>, std::vector<LocalID>, Expression*>>
      worklist;
    size_t index = 0;

    /**
     * Add a work item to build an associated function insider this method, i.e.
     * closure, body of a when clause, local function.
     */
    size_t queue_build_body(
      std::optional<LocalID> receiver,
      std::vector<LocalID> params,
      Expression* body)
    {
      worklist.push({receiver, std::move(params), body});
      index++;
      return index;
    }

    /*
     * The algorithm requires some per basic block metadata, i.e. the map from
     * symbol to SSA variable, the list of pending phi nodes and the "pending"
     * block sets.
     *
     * Since this data won't be used after SSA creation, we store it here in
     * maps rather than as fields of BasicBlock.
     */
    std::map<std::pair<BasicBlock*, LocalID>, Variable> definitions_;

    struct PendingPhi
    {
      LocalID local;
      Variable output;
    };
    std::unordered_map<BasicBlock*, std::vector<PendingPhi>> pending_phis_;
    std::unordered_set<const BasicBlock*> pending_blocks_;

    /**
     * Stack of active scopes.
     *
     * The last element in the vector is the current scope.
     */
    std::vector<std::unique_ptr<ScopeData>> scopes_;

    ScopeData& current_scope()
    {
      assert(!scopes_.empty());
      return *scopes_.back();
    }

    /*
     * The map contains for each block we have visited,
     * the expression that created it and the basic block
     * that is directly before it.
     */
    std::map<BasicBlock*, std::pair<Expression*, BasicBlock*>> original_expr_;

    /*
     * This contains a cache of symbols that are (re)defined by each expression.
     */
    ExprAssignsSymbol assigns_sym_;
  };

  /* Entry point */
  std::unique_ptr<MethodIR> build_ir(const FnSignature& sig, const FnBody& body)
  {
    auto mir = std::make_unique<MethodIR>();
    IRBuilder builder(mir.get());

    std::optional<LocalID> receiver;
    if (sig.receiver)
    {
      receiver = sig.receiver->local;
    }
    std::vector<LocalID> params;
    for (const auto& param : sig.parameters)
    {
      params.push_back(param->local);
    }
    builder.build(receiver, std::move(params), &*body.expression);
    return std::move(mir);
  }
}
