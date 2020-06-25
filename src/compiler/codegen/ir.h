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
      const std::vector<Label>& closure_labels)
    : FunctionGenerator(context, gen, abi),
      reachability_(reachability),
      selectors_(selectors),
      method_(method),
      typecheck_(typecheck),
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
        gen_.define_label(basic_block_label(bb));

        for (const auto& stmt : bb->statements)
        {
          std::visit([&](const auto& s) { visit_stmt(s); }, stmt);
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
    TypeList reify(TypeArgumentsId id)
    {
      return reify(typecheck_.type_arguments.at(id));
    }

    Descriptor entity_descriptor(const Entity* definition, TypeList arguments)
    {
      CodegenItem<Entity> item(definition, Instantiation(arguments));
      return reachability_.find_entity(item).descriptor;
    }

    bytecode::SelectorIdx
    method_selector_index(const std::string& name, TypeList arguments)
    {
      return selectors_.get(Selector::method(name, arguments));
    }

    bytecode::SelectorIdx field_selector_index(const std::string& name)
    {
      return selectors_.get(Selector::field(name));
    }

    void visit_stmt(const CallStmt& stmt)
    {
      bytecode::SelectorIdx selector =
        method_selector_index(stmt.method, reify(stmt.type_arguments));

      FunctionABI abi(stmt);
      allocator_.reserve_child_callspace(abi);

      size_t index = 0;
      emit_copy_to_child(
        abi, Register(truncate<uint8_t>(index++)), variable(stmt.receiver));
      for (const auto& var : stmt.arguments)
      {
        Register reg = variable(var);
        emit_copy_to_child(abi, Register(truncate<uint8_t>(index++)), reg);
      }

      gen_.opcode(Opcode::Call);
      gen_.selector(selector);
      gen_.u8(truncate<uint8_t>(abi.callspace()));

      Register output = variable(stmt.output);
      emit_move_from_child(abi, output, Register(0));
    }

    void visit_stmt(const WhenStmt& stmt)
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
        Register reg = variable(var);
        emit_copy_to_child(abi, Register(truncate<uint8_t>(index++)), reg);
      }
      for (const auto& var : stmt.captures)
      {
        Register reg = variable(var);
        emit_copy_to_child(abi, Register(truncate<uint8_t>(index++)), reg);
      }

      // Gen when opcode with closure
      gen_.opcode(Opcode::When);
      gen_.u32(closure_labels_[stmt.closure_index]);
      gen_.u8(truncate<uint8_t>(stmt.cowns.size()));
      gen_.u8(truncate<uint8_t>(stmt.captures.size()));

      // No output for now TODO-PROMISE
    }

    void visit_stmt(const StaticTypeStmt& stmt)
    {
      Register output = variable(stmt.output);
      Descriptor index =
        entity_descriptor(stmt.definition, reify(stmt.type_arguments));
      emit_load_descriptor(output, index);
    }

    void visit_stmt(const NewStmt& stmt)
    {
      Descriptor index =
        entity_descriptor(stmt.definition, reify(stmt.type_arguments));
      Register descriptor = allocator_.get();
      emit_load_descriptor(descriptor, index);

      Register output = variable(stmt.output);
      if (stmt.parent)
      {
        Register parent = variable(*stmt.parent);
        gen_.opcode(Opcode::NewObject);
        gen_.reg(output);
        gen_.reg(parent);
        gen_.reg(descriptor);
      }
      else
      {
        gen_.opcode(Opcode::NewRegion);
        gen_.reg(output);
        gen_.reg(descriptor);
      }
    }

    void visit_stmt(const MatchBindStmt& stmt)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      emit_copy(output, input);
    }

    void visit_stmt(const ReadFieldStmt& stmt)
    {
      Register base = variable(stmt.base);
      Register output = variable(stmt.output);

      gen_.opcode(Opcode::Load);
      gen_.reg(output);
      gen_.reg(base);
      gen_.selector(field_selector_index(stmt.name));
    }

    void visit_stmt(const WriteFieldStmt& stmt)
    {
      Register base = variable(stmt.base);
      Register output = variable(stmt.output);
      Register right = variable(stmt.right);

      gen_.opcode(Opcode::Store);
      gen_.reg(output);
      gen_.reg(base);
      gen_.selector(field_selector_index(stmt.name));
      gen_.reg(right);
    }

    void visit_stmt(const CopyStmt& stmt)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      emit_copy(output, input);
    }

    void visit_stmt(const IntegerLiteralStmt& stmt)
    {
      Register output = variable(stmt.output);

      gen_.opcode(Opcode::Int64);
      gen_.reg(output);
      gen_.u64(stmt.value);
    }

    void visit_stmt(const StringLiteralStmt& stmt)
    {
      Register output = variable(stmt.output);

      gen_.opcode(Opcode::String);
      gen_.reg(output);
      gen_.str(stmt.value);
    }

    void visit_stmt(const ViewStmt& stmt)
    {
      Register input = variable(stmt.input);
      Register output = variable(stmt.output);
      gen_.opcode(Opcode::MutView);
      gen_.reg(output);
      gen_.reg(input);
    }

    void visit_stmt(const UnitStmt& stmt)
    {
      Register output = variable(stmt.output);
      gen_.opcode(Opcode::Clear);
      gen_.reg(output);
    }

    void visit_stmt(const EndScopeStmt& stmt)
    {
      // TODO: This could be omitted for variables with a non-linear type.
      for (Variable v : stmt.dead_variables)
      {
        gen_.opcode(Opcode::Clear);
        gen_.reg(variable(v));
      }
    }

    void visit_stmt(const OverwriteStmt& stmt)
    {
      gen_.opcode(Opcode::Clear);
      gen_.reg(variable(stmt.dead_variable));
    }

    void visit_term(const BranchTerminator& term)
    {
      const auto& inputs = term.phi_arguments;
      const auto& outputs = term.target->phi_nodes;
      for (auto [in_var, out_var] : safe_zip(inputs, outputs))
      {
        Register input = variable(in_var);
        Register output = variable(out_var);
        emit_move(output, input);
      }

      size_t opcode_start = gen_.current_offset();
      gen_.opcode(Opcode::Jump);
      reference_basic_block(term.target, opcode_start);
    }

    void visit_term(const MatchTerminator& term)
    {
      Register input = variable(term.input);
      Register match_result = allocator_.get();

      for (const auto& arm : term.arms)
      {
        emit_match(match_result, input, arm.type);

        size_t opcode_start = gen_.current_offset();
        gen_.opcode(Opcode::JumpIf);
        gen_.reg(match_result);
        reference_basic_block(arm.target, opcode_start);
      }
      gen_.opcode(Opcode::Unreachable);
    }

    void visit_term(const IfTerminator& term)
    {
      Register input = variable(term.input);

      size_t opcode_start = gen_.current_offset();
      gen_.opcode(Opcode::JumpIf);
      gen_.reg(input);
      reference_basic_block(term.true_target, opcode_start);

      opcode_start = gen_.current_offset();
      gen_.opcode(Opcode::Jump);
      reference_basic_block(term.false_target, opcode_start);
    }

    void visit_term(const ReturnTerminator& term)
    {
      Register input = variable(term.input);

      if (input.index != 0)
      {
        emit_copy(Register(0), input);
        gen_.opcode(Opcode::Clear);
        gen_.reg(input);
      }

      gen_.opcode(Opcode::Return);
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
        it = basic_block_labels_.insert({bb, gen_.create_label()}).first;
      return it->second;
    }

    /**
     * Write a 2-byte relative offset to a basic block.
     *
     * This offset is relative to the start of the current instruction, as
     * specified in `opcode_start`, which would be a few bytes earlier than the
     * current position.
     */
    void reference_basic_block(const BasicBlock* bb, size_t opcode_start)
    {
      gen_.s16(basic_block_label(bb), opcode_start);
    }

    /**
     * Emit instructions that compute the result of matching `input` against
     * `pattern`. The instructions generate an integer value in `output`.
     */
    void emit_match(Register output, Register input, const TypePtr& pattern)
    {
      auto reified_pattern = method_.instantiation.apply(context_, pattern);

      EntityTypePtr entity = reified_pattern->dyncast<EntityType>();
      if (!entity)
      {
        fmt::print(
          std::cerr,
          "Only entity types can be used in pattern matching, found {}\n",
          *reified_pattern);
        abort();
      }

      Descriptor index =
        entity_descriptor(entity->definition, reify(entity->arguments));
      Register descriptor = allocator_.get();
      emit_load_descriptor(descriptor, index);

      gen_.opcode(Opcode::Match);
      gen_.reg(output);
      gen_.reg(input);
      gen_.reg(descriptor);
    }

    const Reachability& reachability_;
    const SelectorTable& selectors_;
    const CodegenItem<Method>& method_;
    const TypecheckResults& typecheck_;
    const std::vector<Label>& closure_labels_;

    FunctionABI abi_ = FunctionABI(*method_.definition->signature);

    std::map<Variable, Register> variables_;
    std::unordered_map<const BasicBlock*, Label> basic_block_labels_;
  };
}
