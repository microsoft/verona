// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "compiler/assignssymbols.h"
#include "compiler/ir/ir.h"
#include "compiler/visitor.h"

#include <queue>
#include <tuple>

/**
 * Build the SSA IR from a function's AST, as a single pass.
 *
 * We use the algorithm from [0] to construct the SSA and the CFG
 * simultaneously.
 *
 * Each basic block maintains a mapping from symbols to SSA variable.
 * This map is updated when processing local variable assignements.
 *
 * When a symbol is referenced, the current BB's map is used to find a
 * definition within the same BB. If the symbol is not found, i.e. it is not
 * defined in this BB, we search for it in all predecessors and create a phi
 * node to merge the definitions.
 *
 * When processing loops, not all predecessors of the loop header are known,
 * preventing the recursion described above. When the loop header block is
 * created, it is marked as "pending". Searching for a symbol in an "pending"
 * block, first checks if the loop defines the variable somewhere.  If it does
 * not then, we can simply skip to find the definition from before the loop.
 * If it does define the variable, then we create a fresh variable, and then
 * record it as a "pending" phi node.
 *
 * Once the loop is 'finish'ed, all predecessors of the header block are known,
 * enabling us to the unset the pending status. For each pending phi node in the
 * header, we search in the predecessors to find definition, and set up a phi
 * node accordingly.
 *
 * [0] describes optimizations to avoid trivial phi nodes, and claims that the
 * resulting SSA is minimal for programs with reducible control flow.
 * So far, we've only implemented the "Marker Algorithm", from
 * Section 3.3 of [0]. This will avoid trivial nodes in acyclic sections of
 * the program, but not when loops are present.
 *
 * We have adapted the marker algorithm, due to the highly structured nature
 * of input language.  By calculating the assignments in an expression, we
 * can generate minimal & pruned SSA.
 *
 * [0]: "Simple and Efficient Construction of Static Single Assignment Form"
 *      https://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
 *
 */

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

  class IRBuilder
  : private ExprVisitor<BuilderResult<IRInput>, ValueKind, BasicBlock*&>
  {
  public:
    static std::unique_ptr<MethodIR>
    build(const FnSignature& sig, const FnBody& body);

  private:
    IRBuilder(MethodIR* ir) : method_ir_(ir)
    {
      scopes_.push_back(std::make_unique<ScopeData>());
    }

    /**
     * Queue building a when expression.  Returns the index where the closure
     * will be added into once complete.
     */
    size_t queue_build(const WhenExpr& when);

    void build(
      std::optional<LocalID> receiver,
      std::vector<LocalID> params,
      Expression* body);

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
    void add_definition(BasicBlock* bb, LocalID local, Variable v);

    Variable find_variable(LocalID local, BasicBlock* bb);

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
    IRInput add_statement(BasicBlock* bb, T stmt);

    /**
     * Add a statement of type T to the given basic block.
     *
     * This method must only be called with statement types that don't have an
     * output.
     */
    template<typename T>
    void add_outputless_statement(BasicBlock* bb, T stmt);

    /**
     * Set the terminator of the given basic block.
     *
     * This method will add to basic block as a predecessor in any basic block
     * the terminator points to.
     */
    template<typename T>
    const T& set_terminator(BasicBlock* bb, T term);

    bool has_pending(const BasicBlock* bb);
    void set_pending(const BasicBlock* bb);
    void finish_block(BasicBlock* bb);

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
      ValueKind kind);

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
      Variable output);

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
      BasicBlock*& bb);

    BuilderResult<IRInput>
    push_scope(Expression& expr, ValueKind kind, BasicBlock*& bb);

    IRInput visit_input(Expression& expr, BasicBlock*& bb);

    BuilderResult<IRInput>
    visit_symbol(SymbolExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_define_local(
      DefineLocalExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_assign_local(
      AssignLocalExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_field(FieldExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_assign_field(
      AssignFieldExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_seq(SeqExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_call(CallExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_match_expr(MatchExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_new_expr(NewExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_empty(EmptyExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_integer_literal_expr(
      IntegerLiteralExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_string_literal_expr(
      StringLiteralExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_when(WhenExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_while(WhileExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_if(IfExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_block(BlockExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput>
    visit_view_expr(ViewExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    BuilderResult<IRInput> visit_binary_operator_expr(
      BinaryOperatorExpr& expr, ValueKind kind, BasicBlock*& bb) final;

    /**
     * Generate a unit value, if needed.
     *
     * If kind is Unused, BuilderResult::Invalid is returned. Otherwise a fresh
     * variable is created and a UnitStmt is generated to write to it.
     */
    BuilderResult<IRInput> unit(
      SourceManager::SourceRange source_range, ValueKind kind, BasicBlock* bb);

    /**
     * For each basic block `term` jumps to, add `pred` as a predecessor.
     */
    void
    add_terminator_predecessor(BasicBlock* pred, const ReturnTerminator& term);
    void
    add_terminator_predecessor(BasicBlock* pred, const BranchTerminator& term);
    void
    add_terminator_predecessor(BasicBlock* pred, const MatchTerminator& term);
    void add_terminator_predecessor(BasicBlock* pred, const IfTerminator& term);

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
}
