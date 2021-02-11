// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/infer.h"

#include "compiler/ast.h"
#include "compiler/format.h"
#include "compiler/instantiation.h"
#include "compiler/printing.h"
#include "compiler/typecheck/constraint.h"
#include "compiler/typecheck/solver.h"
#include "compiler/zip.h"
#include "ds/error.h"

#include <fmt/ostream.h>
#include <iostream>
#include <set>

namespace verona::compiler
{
  class Infer
  {
  public:
    Infer(
      Context& context,
      const Program& program,
      const Method& method,
      const LivenessAnalysis& liveness,
      InferResults* results,
      const MethodIR& mir)
    : context_(context),
      program_(program),
      method_(method),
      liveness_(liveness),
      results_(results),
      mir_(mir)

    {}

    void process(const FunctionIR& ir)
    {
      IRTraversal traversal(ir);
      while (const BasicBlock* bb = traversal.next())
      {
        TypeAssignment& assignment = results_->types[bb];

        setup_basic_block(assignment, bb);

        std::vector<Variable> dead_variables;
        for (const auto& stmt : bb->statements)
        {
          std::visit(
            [&](const auto& inner) {
              visit_stmt(assignment, dead_variables, inner);
            },
            stmt);
        }

        const Terminator& term = bb->terminator.value();
        std::visit([&](const auto& t) { visit_term(assignment, t); }, term);

        finish_basic_block(assignment, dead_variables, bb);
      }
      // All subsequent calls are for closures.
      closure_ = true;
    }

    void set_parameter_types(const FunctionIR& ir)
    {
      TypeAssignment& assignment = results_->types[ir.entry];

      RewriteRegions rewrite_regions(context_, ir);

      const TypeSignature& signature = method_.signature->types;
      if (ir.receiver)
      {
        TypePtr self = self_type(method_.parent);
        TypePtr receiver_type =
          context_.mk_intersection(self, signature.receiver);

        set_type(
          assignment, *ir.receiver, rewrite_regions.apply(receiver_type));
      }

      for (const auto& [param, var] :
           safe_zip(signature.arguments, ir.parameters))
      {
        set_type(assignment, var, rewrite_regions.apply(param));
      }
    }

  private:
    TypePtr self_type(const Entity* entity)
    {
      TypeList arguments;
      for (const auto& param : entity->generics->types)
      {
        arguments.push_back(
          context_.mk_type_parameter(param.get(), TypeParameter::Expanded::No));
      }
      return context_.mk_entity_type(entity, arguments);
    }

    /**
     * Replace Receiver and Parameter regions that appear in the signature to
     * Variable regions, based on what SSA variable the receiver and parameters
     * got bound to.
     */
    struct RewriteRegions : public RecursiveTypeMapper
    {
      RewriteRegions(Context& context, const FunctionIR& ir)
      : RecursiveTypeMapper(context), ir_(ir)
      {}

      Region visit_region(const Region& region)
      {
        return match(
          region,
          [&](const RegionNone& r) -> Region { return r; },
          [&](const RegionHole& r) -> Region { return r; },
          [&](const RegionExternal& r) -> Region { return r; },
          [&](const RegionReceiver& r) -> Region {
            return RegionVariable{*ir_.receiver};
          },
          [&](const RegionParameter& r) -> Region {
            return RegionVariable{ir_.parameters.at(r.index)};
          },
          [&](const RegionVariable& r) -> Region { abort(); });
      }

      TypePtr visit_capability(const CapabilityTypePtr& ty)
      {
        return context().mk_capability(ty->kind, visit_region(ty->region));
      }

      TypePtr visit_apply_region(const ApplyRegionTypePtr& ty)
      {
        return context().mk_apply_region(
          ty->mode, visit_region(ty->region), apply(ty->type));
      }

      const FunctionIR& ir_;
    };

    /**
     * Setup types of incoming variables for a basic block.
     *
     * This defines types for live-in variables, by taking the union of the
     * types from predecessor blocks. The types all have a VariableRenaming
     * applied to them, to map any Phi-renamed regions.
     */
    void setup_basic_block(TypeAssignment& assignment, const BasicBlock* bb)
    {
      if (bb->predecessors.empty())
      {
        // If there are no predecessors, then this is the entrypoint of the IR.
        // The only live variables are the method parameters, and their types
        // have already been defined by `set_parameter_types`.
        const Liveness& liveness_in = liveness_.state_in(bb);
        for (Variable variable : liveness_in.live_variables)
        {
          assert(assignment.find(variable) != assignment.end());
        }
        for (Variable variable : liveness_in.zombie_variables)
        {
          assert(assignment.find(variable) != assignment.end());
        }
        return;
      }

      std::vector<VariableRenaming> forward_renamings;
      std::vector<VariableRenaming> backward_renamings;
      for (const BasicBlock* predecessor : bb->predecessors)
      {
        forward_renamings.push_back(
          VariableRenaming::forwards(predecessor, bb));

        backward_renamings.push_back(
          VariableRenaming::backwards(predecessor, bb));
      }

      /**
       * Get the type at the exit of a basic block.
       * If there is no type known yet because we haven't visited that block yet
       * (i.e. this is a back edge), we assign it a fresh inference variable.
       *
       * `finish_basic_block` on the predecessor will later add the constraint
       * between that inference variable and the actual type.
       */
      auto get_exit_type = [&](const BasicBlock* predecessor, Variable v) {
        auto [it, inserted] = exit_types_[predecessor].insert({v, nullptr});
        if (inserted)
          it->second = fresh_type_var();
        return it->second;
      };

      /**
       * Look up the type of the variable in the predecessor (possibly using the
       * renaming to follow Phi nodes), and propagate the type in this block.
       */
      auto propagate_variable = [&](Variable variable) {
        std::set<TypePtr> types;
        for (size_t j = 0; j < bb->predecessors.size(); j++)
        {
          const BasicBlock* predecessor = bb->predecessors.at(j);

          Variable input = backward_renamings.at(j).apply(variable);

          TypePtr incoming_type = get_exit_type(predecessor, input);
          TypePtr renamed_type = context_.mk_variable_renaming(
            forward_renamings.at(j), incoming_type);
          types.insert(renamed_type);
        }

        set_type(assignment, variable, context_.mk_union(types));
      };

      const Liveness& liveness_in = liveness_.state_in(bb);
      for (Variable variable : liveness_in.live_variables)
      {
        propagate_variable(variable);
      }
      for (Variable variable : liveness_in.zombie_variables)
      {
        propagate_variable(variable);
      }
    }

    /**
     * Given the list of variables that were killed by a basic block, compute
     * the PathCompressionMap used to erase those variable names from the types
     * of live-out variables of the block.
     */
    PathCompressionMap build_compression_map(
      const BasicBlock* bb, const std::vector<Variable>& dead_variables)
    {
      PathCompressionMap compression;
      for (Variable v : dead_variables)
      {
        // The path compression operator already applies to itself,
        // e.g. in compress([x: T, y: U], mut(y)), U is allowed to refer to
        // (x: T).
        //
        // This lets us have multiple variables go out of scope at the same
        // time, where the variable and their types form a DAG.
        //
        compression.insert({v, context_.mk_indirect_type(bb, v)});
      }
      return compression;
    }

    /**
     * Copies the types of live-out variables from `assignment` into
     * `exit_types_`.
     */
    void finish_basic_block(
      const TypeAssignment& assignment,
      const std::vector<Variable>& dead_variables,
      const BasicBlock* bb)
    {
      PathCompressionMap compression =
        build_compression_map(bb, dead_variables);

      TypeAssignment& types = exit_types_[bb];
      auto assign_type = [&](Variable v) {
        TypePtr compressed_type =
          context_.mk_path_compression(compression, assignment.at(v));

        // If the basic block's exit types haven't been used yet we can just
        // set them to the types from the assignment.
        auto [it, inserted] = types.insert({v, compressed_type});

        // If the exit type has already been used (i.e. this is a loop body),
        // they would have been set to an inference variables. We can't redefine
        // the type so we add a constraint between this inference variables and
        // the type we got out of traversing the BB.
        if (!inserted)
          add_constraint(compressed_type, it->second, "finish_basic_block");
      };

      const Liveness& liveness_out = liveness_.state_out(bb);
      for (Variable variable : liveness_out.live_variables)
      {
        assign_type(variable);
      }
      for (Variable variable : liveness_out.zombie_variables)
      {
        assign_type(variable);
      }
      for (Variable variable : dead_variables)
      {
        assert(!liveness_out.live_variables.contains(variable));
        assert(!liveness_out.zombie_variables.contains(variable));
      }
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const CallStmt& stmt)
    {
      InferableTypeSequence tyargs = fresh_unbounded_sequence();

      TypePtr receiver = get_type(assignment, stmt.receiver);
      TypeList arguments = get_types(assignment, stmt.arguments);
      TypePtr result = fresh_type_var();

      TypeSignature signature(receiver, arguments, result);

      add_constraint(
        receiver,
        context_.mk_has_applied_method(stmt.method, tyargs, signature),
        "call stmt");
      set_type(assignment, stmt.output, result);
      set_type_arguments(stmt.type_arguments, tyargs);
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const WhenStmt& stmt)
    {
      auto& closure = *mir_.function_irs[stmt.closure_index];

      TypeAssignment& closure_assignment = results_->types[closure.entry];

      // Link cown parameters
      for (size_t i = 0; i < stmt.cowns.size(); i++)
      {
        const Variable& cown = stmt.cowns[i];
        TypePtr contents = context_.mk_entity_of(fresh_type_var());
        TypePtr cown_type = get_entity("cown", {contents});
        TypePtr expected_type =
          context_.mk_intersection(cown_type, context_.mk_immutable());

        add_constraint(get_type(assignment, cown), expected_type, "cown_param");

        const auto& param = closure.parameters[i];
        Region region = RegionExternal{
          i}; // Really i, what about multiple whens? PAUL TO COMMENT
        TypePtr cap = context_.mk_mutable(region);
        set_type(
          closure_assignment, param, context_.mk_intersection(contents, cap));
      }

      // Link capture parameter types
      for (size_t i = 0; i < stmt.captures.size(); i++)
      {
        const auto& capture = stmt.captures[i];
        TypePtr capt_typ = get_type(assignment, capture);

        const auto& param = closure.parameters[i + stmt.cowns.size()];
        set_type(closure_assignment, param, capt_typ);
      }

      set_type(assignment, stmt.output, context_.mk_unit());
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const StaticTypeStmt& stmt)
    {
      TypeList arguments = fresh_entity_type_arguments(stmt.definition);
      TypePtr result = context_.mk_static_type(stmt.definition, arguments);

      set_type(assignment, stmt.output, result);
      set_type_arguments(stmt.type_arguments, BoundedTypeSequence(arguments));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const NewStmt& stmt)
    {
      TypePtr cap;
      if (stmt.parent)
      {
        Region region = RegionVariable{*stmt.parent};
        cap = context_.mk_mutable(region);
      }
      else
      {
        cap = context_.mk_isolated(RegionNone{});
      }

      TypeList arguments = fresh_entity_type_arguments(stmt.definition);
      TypePtr entity = context_.mk_entity_type(stmt.definition, arguments);

      set_type(assignment, stmt.output, context_.mk_intersection(cap, entity));
      set_type_arguments(stmt.type_arguments, BoundedTypeSequence(arguments));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const MatchBindStmt& stmt)
    {
      TypePtr input = get_type(assignment, stmt.input);
      set_type(
        assignment, stmt.output, context_.mk_intersection(stmt.type, input));
    }

    TypePtr view_field(const TypePtr& base, std::string name)
    {
      // TODO: This could be special cased for times when we know the base type
      // already, avoiding the need for a new inference variable and constraint.
      TypePtr type = fresh_type_var();
      add_constraint(base, context_.mk_delayed_field_view(name, type), "field");
      return type;
    }

    /**
     * Get the read and write types for a field in the given base.
     */
    std::pair<TypePtr, TypePtr>
    lookup_field(const TypePtr& base, std::string name)
    {
      // TODO: This could be special cased for times when we know the base type
      // already, avoiding the need for a new inference variable and constraint.

      // We don't want to apply a view when looking up the field, so we use mut
      // as an identity value. The region doesn't really matter, so we just use
      // the hole.
      TypePtr view = context_.mk_mutable(RegionHole{});
      TypePtr read = fresh_type_var();
      TypePtr write = fresh_type_var();
      add_constraint(
        base, context_.mk_has_field(view, name, read, write), "lookup_field");
      return {read, write};
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const ReadFieldStmt& stmt)
    {
      TypePtr base = get_type(assignment, stmt.base);
      TypePtr field = view_field(base, stmt.name);

      Region region = RegionVariable{stmt.base};
      set_type(
        assignment,
        stmt.output,
        context_.mk_apply_region(ApplyRegionType::Mode::Adapt, region, field));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const WriteFieldStmt& stmt)
    {
      TypePtr base = get_type(assignment, stmt.base);
      TypePtr right = get_type(assignment, stmt.right);

      auto [field_read, field_write] = lookup_field(base, stmt.name);

      Region region = RegionVariable{stmt.base};
      add_constraint(
        right,
        context_.mk_apply_region(
          ApplyRegionType::Mode::Extract, region, field_write),
        "WriteFieldStmt");

      set_type(
        assignment,
        stmt.output,
        context_.mk_apply_region(
          ApplyRegionType::Mode::Extract, region, field_read));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const CopyStmt& stmt)
    {
      set_type(assignment, stmt.output, get_type(assignment, stmt.input));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const IntegerLiteralStmt& stmt)
    {
      TypePtr u64 = get_entity("U64");
      TypePtr type = context_.mk_intersection(u64, context_.mk_immutable());

      set_type(assignment, stmt.output, type);
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const StringLiteralStmt& stmt)
    {
      TypePtr u64 = get_entity("String");
      TypePtr type = context_.mk_intersection(u64, context_.mk_immutable());
      set_type(assignment, stmt.output, type);
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const UnitStmt& stmt)
    {
      set_type(assignment, stmt.output, context_.mk_unit());
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const ViewStmt& stmt)
    {
      TypePtr input = get_type(assignment, stmt.input);
      TypePtr entity = context_.mk_entity_of(input);

      Region region = RegionVariable{stmt.input};
      TypePtr cap = context_.mk_mutable(region);

      set_type(assignment, stmt.output, context_.mk_intersection(entity, cap));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const EndScopeStmt& stmt)
    {
      std::copy(
        stmt.dead_variables.begin(),
        stmt.dead_variables.end(),
        std::back_inserter(dead_variables));
    }

    void visit_stmt(
      TypeAssignment& assignment,
      std::vector<Variable>& dead_variables,
      const OverwriteStmt& stmt)
    {
      dead_variables.push_back(stmt.dead_variable);
    }

    void visit_term(TypeAssignment& assignment, const BranchTerminator& term) {}

    void visit_term(TypeAssignment& assignment, const ReturnTerminator& term)
    {
      if (!closure_)
        add_constraint(
          get_type(assignment, term.input),
          method_.signature->types.return_type,
          "return_term");
      else
        add_constraint(
          get_type(assignment, term.input),
          context_.mk_unit(),
          "return_closure_term");
    }

    void visit_term(TypeAssignment& assignment, const MatchTerminator& term)
    {
      // TODO: check for exhaustiveness
    }

    void visit_term(TypeAssignment& assignment, const IfTerminator& term)
    {
      TypePtr input = get_type(assignment, term.input);
      TypePtr u64 = get_entity("U64");
      TypePtr expected = context_.mk_intersection(u64, context_.mk_immutable());

      add_constraint(input, expected, "if_term");
    }

    void set_type(TypeAssignment& assignment, Variable v, TypePtr ty)
    {
      auto it = assignment.insert({v, ty});
      if (!it.second)
      {
        add_constraint(ty, it.first->second, "set_type");
      }
    }

    void set_type_arguments(TypeArgumentsId id, InferableTypeSequence types)
    {
      auto it = results_->type_arguments.insert({id, types});
      if (!it.second)
      {
        InternalError::print("TypeArguments already exist\n");
      }
    }

    TypePtr get_type(const TypeAssignment& assignment, Variable v)
    {
      return assignment.at(v);
    }

    TypeList
    get_types(TypeAssignment& assignment, const std::vector<IRInput>& variables)
    {
      TypeList types;
      for (const auto& v : variables)
      {
        types.push_back(get_type(assignment, v));
      }
      return types;
    }

    void add_constraint(TypePtr sub, TypePtr super, std::string reason)
    {
      results_->constraints.emplace_back(sub, super, 0, context_, reason);
    }

    TypePtr fresh_type_var()
    {
      uint64_t index = next_fresh_type_var_++;
      return context_.mk_infer_range(index, std::nullopt);
    }

    UnboundedTypeSequence fresh_unbounded_sequence()
    {
      uint64_t index = next_fresh_type_var_++;
      return UnboundedTypeSequence(index);
    }

    /**
     * Generate new fresh type arguments to apply to an entity.
     */
    TypeList fresh_entity_type_arguments(const Entity* entity)
    {
      const Generics& generics = *entity->generics;

      TypeList types;
      for (const auto& param : generics.types)
      {
        TypePtr ty = fresh_type_var();
        types.push_back(ty);
      }

      // Require the type arguments to satisfy the bounds.
      //
      // We need to substitute the type arguments in the bounds themselves
      // (using the Instantiation) in order to support F-bounded polymorphism.
      Instantiation instantiation(types);
      for (const auto& [param, ty] : safe_zip(generics.types, types))
      {
        add_constraint(ty, instantiation.apply(context_, param->bound), "args");
      }

      return types;
    }

    /**
     * Get an entity type by name.
     *
     * This is used to locate standard library classes which the compiler has
     * special knowledge of, eg. U64 or Cown.
     */
    TypePtr get_entity(const std::string& name, TypeList args = TypeList())
    {
      const Entity* entity = program_.find_entity(name);
      if (!entity)
      {
        abort();
      }
      return context_.mk_entity_type(entity, args);
    }

    Context& context_;
    const Program& program_;
    const Method& method_;
    const LivenessAnalysis& liveness_;
    InferResults* results_;
    const MethodIR& mir_;
    bool closure_ = false;

    uint64_t next_fresh_type_var_ = 0;

    /**
     * Types of live variables at the exit of a BasicBlock. The types have path
     * compression applied already.
     */
    std::unordered_map<const BasicBlock*, TypeAssignment> exit_types_;
  };

  std::unique_ptr<InferResults> infer(
    Context& context,
    const Program& program,
    const Method& method,
    const MethodIR& mir,
    const LivenessAnalysis& liveness)
  {
    auto results = std::make_unique<InferResults>();
    Infer inferer(context, program, method, liveness, results.get(), mir);
    inferer.set_parameter_types(*mir.function_irs.front());
    for (auto& ir : mir.function_irs)
      inferer.process(*ir);
    results->dump(context, method);
    return results;
  }

  void InferResults::dump(Context& context, const Method& method)
  {
    std::string path = method.path();

    fmt::print(
      *context.dump(path, "constraints"),
      "Constraints for {}\n{}\n",
      path,
      format::lines(constraints));

    dump_types(context, method, "infer", "Infer Types", types);
  }

  void dump_types(
    Context& context,
    const Method& method,
    std::string_view name,
    std::string_view title,
    const std::unordered_map<const BasicBlock*, TypeAssignment>& types)
  {
    using format::lines;
    using format::sorted;

    std::string path = method.path();
    auto output = context.dump(path, name);

    fmt::print(*output, "{} for {}:\n", title, path);
    for (const auto& [bb, assignment] : types)
    {
      auto format_entry = [](const auto& entry) {
        return fmt::format("    {}: {}", entry.first, *entry.second);
      };

      fmt::print(
        *output,
        "  Basic block {}:\n{}\n",
        *bb,
        lines(sorted(assignment, format_entry)));
    }
  }
}
