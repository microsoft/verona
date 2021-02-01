// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/function.h"
#include "compiler/ir/ir.h"
#include "compiler/printing.h"
#include "compiler/typecheck/typecheck.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  using bytecode::Opcode;
  using bytecode::SelectorIdx;

  /**
   * Class for generating the body of methods from their IR.
   */
  class IRGenerator : public FunctionGenerator
  {
  public:
    IRGenerator(
      Context& context,
      const Reachability& reachability,
      const SelectorTable& selectors,
      Generator& gen,
      FunctionABI abi,
      const CodegenItem<Method>& method,
      const TypecheckResults& typecheck,
      const LivenessAnalysis& liveness,
      const std::vector<Label>& closure_labels)
    : FunctionGenerator(context, gen, abi),
      reachability_(reachability),
      selectors_(selectors),
      method_(method),
      typecheck_(typecheck),
      liveness_(liveness),
      closure_labels_(closure_labels)
    {}

    void generate_body(const FunctionIR& ir)
    {
      setup_parameters(ir);

      IRTraversal traversal(ir);
      // IRTraversal always returns the entrypoint first,
      // which is what we want here.
      while (BasicBlock* bb = traversal.next())
      {
        define_label(basic_block_label(bb));

        std::vector<Liveness> live_out = liveness_.statements_out(bb);
        for (const auto& [stmt, stmt_live_out] :
             safe_zip(bb->statements, live_out))
        {
          const auto& stmt_live_out_ = stmt_live_out;
          std::visit(
            [&](const auto& s) { visit_stmt(s, stmt_live_out_); }, stmt);
        }

        const Terminator& term = bb->terminator.value();
        std::visit([&](const auto& t) { visit_term(t); }, term);
      }
    }

  private:
    void setup_parameters(const FunctionIR& ir)
    {
      if (const auto& receiver = ir.receiver)
      {
        variables_.insert({*receiver, Register(0)});
      }

      size_t index = 1;
      for (const auto& arg : ir.parameters)
      {
        variables_.insert({arg, Register(truncate<uint8_t>(index++))});
      }
    }

    /**
     * Map some list of types to concrete types, using the current method's
     * instantiation.
     */
    TypeList reify(const TypeList& arguments)
    {
      return method_.instantiation.apply(context_, arguments);
    }
    TypePtr reify(const TypePtr& type)
    {
      return method_.instantiation.apply(context_, type);
    }
    TypeList reify(TypeArgumentsId id)
    {
      return reify(typecheck_.type_arguments.at(id));
    }

    Descriptor entity_descriptor(const Entity* definition, TypeList arguments)
    {
      CodegenItem<Entity> item(definition, Instantiation(arguments));
      return reachability_.find_entity(item).descriptor;
    }

    SelectorIdx method_selector(const std::string& name, TypeList arguments)
    {
      return selectors_.get(Selector::method(name, arguments));
    }

    SelectorIdx field_selector(const std::string& name)
    {
      return selectors_.get(Selector::field(name));
    }

    /**
     * Use a pair of PROTECT/UNPROTECT calls to prevent live variables from
     * being garbage collected for the duration of a CALL statement.
     *
     * The `fn` argument will be executed in between the emission of the two
     * opcodes, and is used to emit the actual CALL opcode.
     */
    template<typename Fn>
    void protect_live_registers(
      const CallStmt& stmt, const Liveness& live_out, Fn&& fn)
    {
      std::vector<Register> registers;
      for (const Variable& var : live_out.live_variables)
      {
        // We care about variables that are live *during* the call, but we're
        // iterating over those that are live *after* it. Thankfully the only
        // difference is in the output value of the call statement, so we skip
        // that.
        if (var != stmt.output)
        {
          registers.push_back(variable(var));
        }
      }

      if (!registers.empty())
      {
        emit<Opcode::Protect>(registers);
      }

      std::forward<Fn>(fn)();

      if (!registers.empty())
      {
        emit<Opcode::Unprotect>(registers);
      }
    }

    void visit_stmt(const CallStmt& stmt, const Liveness& live_out)
    {
      SelectorIdx selector =
        method_selector(stmt.method, reify(stmt.type_arguments));

      FunctionABI abi(stmt);
      allocator_.reserve_child_callspace(abi);

      size_t index = 0;

      emit<Opcode::Copy>(
        callee_register(abi, truncate<uint8_t>(index++)),
        variable(stmt.receiver));
      for (const auto& var : stmt.arguments)
      {
        Register src = variable(var);
        CalleeRegister dst = callee_register(abi, truncate<uint8_t>(index++));
        emit<Opcode::Copy>(dst, src);
      }

      protect_live_registers(stmt, live_out, [&]() {
        emit<Opcode::Call>(selector, truncate<uint8_t>(abi.callspace()));
      });

      emit<Opcode::Move>(variable(stmt.output), callee_register(abi, 0));
    }

    void visit_stmt(const WhenStmt& stmt, const Liveness& live_out)
    {
      // No clever type params on When closures yet.
      TypeList empty;

      FunctionABI abi(stmt);
      allocator_.reserve_child_callspace(abi);

      // Store When arguments onto stack
      // Index is 1, as we don't have a receiver (static method),
      // but currently codegen always assumes a receiver.
      // TODO-Better-Static-codegen
      size_t index = 1;
      for (const auto& var : stmt.cowns)
      {
        Register src = variable(var);
        CalleeRegister dst = callee_register(abi, truncate<uint8_t>(index++));
        emit<Opcode::Copy>(dst, src);
      }
      for (const auto& var : stmt.captures)
      {
        Register src = variable(var);
        CalleeRegister dst = callee_register(abi, truncate<uint8_t>(index++));
        emit<Opcode::Copy>(dst, src);
      }

      // Gen when opcode with closure
      emit<Opcode::When>(
        closure_labels_[stmt.closure_index],
        truncate<uint8_t>(stmt.cowns.size()),
        truncate<uint8_t>(stmt.captures.size()));

      // No output for now TODO-PROMISE
    }

    void visit_stmt(const StaticTypeStmt& stmt, const Liveness& live_out)
    {
      Register output = variable(stmt.output);
      Descriptor index =
        entity_descriptor(stmt.definition, reify(stmt.type_arguments));
      emit<Opcode::LoadDescriptor>(output, index);
    }

    void visit_stmt(const NewStmt& stmt, const Liveness& live_out)
    {
      Descriptor index =
        entity_descriptor(stmt.definition, reify(stmt.type_arguments));

      Register descriptor = allocator_.get();
      emit<Opcode::LoadDescriptor>(descriptor, index);

      Register output = variable(stmt.output);
      if (stmt.parent)
      {
        Register parent = variable(*stmt.parent);
        emit<Opcode::NewObject>(output, parent, descriptor);
      }
      else
      {
        emit<Opcode::NewRegion>(output, descriptor);
      }
    }

    void visit_stmt(const MatchBindStmt& stmt, const Liveness& live_out)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      emit<Opcode::Copy>(output, input);
    }

    void visit_stmt(const ReadFieldStmt& stmt, const Liveness& live_out)
    {
      Register base = variable(stmt.base);
      Register output = variable(stmt.output);
      SelectorIdx selector = field_selector(stmt.name);

      emit<Opcode::Load>(output, base, selector);
    }

    void visit_stmt(const WriteFieldStmt& stmt, const Liveness& live_out)
    {
      Register base = variable(stmt.base);
      Register output = variable(stmt.output);
      Register right = variable(stmt.right);
      SelectorIdx selector = field_selector(stmt.name);

      emit<Opcode::Store>(output, base, selector, right);
    }

    void visit_stmt(const CopyStmt& stmt, const Liveness& live_out)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      emit<Opcode::Copy>(output, input);
    }

    void visit_stmt(const IntegerLiteralStmt& stmt, const Liveness& live_out)
    {
      Register output = variable(stmt.output);
      emit<Opcode::Int64>(output, stmt.value);
    }

    void visit_stmt(const StringLiteralStmt& stmt, const Liveness& live_out)
    {
      Register output = variable(stmt.output);
      emit<Opcode::String>(output, stmt.value);
    }

    void visit_stmt(const ViewStmt& stmt, const Liveness& live_out)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      emit<Opcode::MutView>(output, input);
    }

    void visit_stmt(const UnitStmt& stmt, const Liveness& live_out)
    {
      Register output = variable(stmt.output);
      emit<Opcode::Clear>(output);
    }

    void visit_stmt(const EndScopeStmt& stmt, const Liveness& live_out)
    {
      std::vector<Register> regs;
      // TODO: This could be omitted for variables with a non-linear type.
      for (Variable v : stmt.dead_variables)
      {
        regs.push_back(variable(v));
      }

      if (!regs.empty())
      {
        emit<Opcode::ClearList>(regs);
      }
    }

    void visit_stmt(const OverwriteStmt& stmt, const Liveness& live_out)
    {
      emit<Opcode::Clear>(variable(stmt.dead_variable));
    }

    void visit_term(const BranchTerminator& term)
    {
      const auto& inputs = term.phi_arguments;
      const auto& outputs = term.target->phi_nodes;
      for (auto [in_var, out_var] : safe_zip(inputs, outputs))
      {
        emit<Opcode::Move>(variable(out_var), variable(in_var));
      }

      emit<Opcode::Jump>(basic_block_label(term.target));
    }

    void visit_term(const MatchTerminator& term)
    {
      Register input = variable(term.input);
      Register match_result = allocator_.get();

      for (const auto& arm : term.arms)
      {
        TypePtr reified_pattern = reify(arm.type);
        EmitMatch(this, input, context_)
          .visit_type(reified_pattern, match_result);
        emit<Opcode::JumpIf>(match_result, basic_block_label(arm.target));
      }
      emit<Opcode::Unreachable>();
    }

    void visit_term(const IfTerminator& term)
    {
      Register input = variable(term.input);
      emit<Opcode::JumpIf>(input, basic_block_label(term.true_target));
      emit<Opcode::Jump>(basic_block_label(term.false_target));
    }

    void visit_term(const ReturnTerminator& term)
    {
      Register input = variable(term.input);

      if (input.value != 0)
      {
        emit<Opcode::Copy>(Register(0), input);
        emit<Opcode::Clear>(input);
      }

      emit<Opcode::Return>();
    }

    /**
     * Get the Register associated with the given SSA variable.
     *
     * Registers are allocated lazily when this function is first called for the
     * given basic block. Registers are currently never reused by other
     * variables.
     */
    Register variable(Variable var)
    {
      auto [it, inserted] = variables_.insert({var, Register(0)});
      if (inserted)
      {
        it->second = allocator_.get();
      }
      return it->second;
    }

    /**
     * Get the Label associated with the given basic block's address.
     *
     * Labels are created lazily when this function is first called for the
     * given basic block.
     */
    Label basic_block_label(const BasicBlock* bb)
    {
      // TODO: Avoid the double lookup. It's annoying to do because Label does
      // not have a dummy value.
      auto it = basic_block_labels_.find(bb);
      if (it == basic_block_labels_.end())
        it = basic_block_labels_.insert({bb, create_label()}).first;
      return it->second;
    }

    /**
     * Type visitor used to emit the right bytecode sequence to match a value
     * against a given type.
     *
     * The visitor takes as an additional argument the register in which the
     * value being matched on is located. It returns a register which holds the
     * boolean result.
     */
    struct EmitMatch : public TypeVisitor<void, Register>
    {
      EmitMatch(IRGenerator* parent, Register input, Context& context)
      : parent(parent), input(input), context_(context)
      {}

      void
      visit_entity_type(const EntityTypePtr& entity, Register output) override
      {
        Descriptor index =
          parent->entity_descriptor(entity->definition, entity->arguments);
        Register descriptor = parent->allocator_.get();
        parent->emit<Opcode::LoadDescriptor>(descriptor, index);
        parent->emit<Opcode::MatchDescriptor>(output, input, descriptor);
      }

      void visit_capability(
        const CapabilityTypePtr& capability, Register output) override
      {
        switch (capability->kind)
        {
          case CapabilityKind::Isolated:
            assert(std::holds_alternative<RegionHole>(capability->region));
            parent->emit<Opcode::MatchCapability>(
              output, input, bytecode::Capability::Iso);
            break;

          case CapabilityKind::Immutable:
            assert(std::holds_alternative<RegionNone>(capability->region));
            parent->emit<Opcode::MatchCapability>(
              output, input, bytecode::Capability::Imm);
            break;

          case CapabilityKind::Mutable:
            assert(std::holds_alternative<RegionHole>(capability->region));
            parent->emit<Opcode::MatchCapability>(
              output, input, bytecode::Capability::Mut);
            break;

          case CapabilityKind::Subregion:
            abort();
        }
      }

      void visit_union(const UnionTypePtr& type, Register output) override
      {
        emit_connective_match(
          output, type->elements, 0, bytecode::BinaryOperator::Or);
      }

      void visit_intersection(
        const IntersectionTypePtr& type, Register output) override
      {
        emit_connective_match(
          output, type->elements, 0, bytecode::BinaryOperator::And);
      }

      void visit_base_type(const TypePtr& type, Register input) override
      {
        // TODO: Ultimately, this should be a non-user facing error.
        // InternalError::print(
        //   "Matching against type {} is not supported\n", *type);
        // However, currently the earlier phases do not catch this.
        report(
          context_,
          std::nullopt,
          DiagnosticKind::Error,
          Diagnostic::PatternMatchOnUnsupportedType,
          type);
      }

      /**
       * Emit the match code for a union or intersection type. This matches the
       * input against every elements of the connective, and combines the
       * results using `op`. If the connective is empty, we directly produce
       * `identity` as the result.
       *
       * TODO: this could generate more efficient code by short-circuting the
       * process. For instance when matching on (A & B), if the A match fails
       * there is no point in trying to match against B.
       *
       * Additionally this makes very inefficient use of register allocation,
       * just like the rest of the code generator.
       */
      void emit_connective_match(
        Register output,
        const TypeSet& elements,
        uint64_t identity,
        bytecode::BinaryOperator op)
      {
        if (elements.empty())
        {
          parent->emit<Opcode::Int64>(output, identity);
        }
        else
        {
          // Unions and intersections are always normalised such that they are
          // elided when they only have a single component.
          assert(elements.size() > 1);

          // We evaluate the first element and place the result in `output`.
          // We then evaluate each subsequent element into `rhs`, and fold the
          // results into `output` using the specified boolean operator.
          auto it = elements.begin();
          visit_type(*it, output);
          it++;

          Register rhs = parent->allocator_.get();
          for (; it != elements.end(); it++)
          {
            visit_type(*it, rhs);

            parent->emit<Opcode::BinOp>(output, op, output, rhs);
          }
        }
      }

    private:
      IRGenerator* parent;
      Register input;
      Context& context_;
    };

    const Reachability& reachability_;
    const SelectorTable& selectors_;
    const CodegenItem<Method>& method_;
    const TypecheckResults& typecheck_;
    const LivenessAnalysis& liveness_;
    const std::vector<Label>& closure_labels_;

    FunctionABI abi_ = FunctionABI(*method_.definition->signature);

    std::map<Variable, Register> variables_;
    std::unordered_map<const BasicBlock*, Label> basic_block_labels_;
  };
}
