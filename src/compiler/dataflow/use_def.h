// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/variable_set.h"
#include "compiler/ir/ir.h"

namespace verona::compiler
{
  /**
   * UseDefVisitor allows analysis of use and definition points of the IR.
   *
   * It will walk a basic block backwards, starting from the terminator until
   * the Phi nodes. The visitor will call the `use_variable`, `define_variable`,
   * `phi_inputs` and `phi_outputs` methods on the result value as it encounter
   * these points.
   *
   * It can also be used on individual statements and terminators, allowing the
   * caller to execute code at each step. See `IncrementalLiveness` for an
   * example.
   */
  template<typename Result>
  class UseDefVisitor
  {
  public:
    UseDefVisitor(Result& result) : result_(result) {}

    void visit_basic_block(const BasicBlock* bb)
    {
      visit_term(bb->terminator.value());

      for (auto it = bb->statements.rbegin(); it != bb->statements.rend(); it++)
      {
        visit_stmt(*it);
      }

      result_.phi_outputs(bb->phi_nodes);
    }

    void visit_stmt(const Statement& stmt)
    {
      std::visit([&](const auto& inner) { visit_inner_stmt(inner); }, stmt);
    }

    void visit_term(const Terminator& term)
    {
      std::visit([&](const auto& t) { visit_inner_term(t); }, term);
    }

  private:
    void visit_inner_stmt(const CallStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.receiver);
      use_variables(stmt.arguments);
    }

    void visit_inner_stmt(const WhenStmt& stmt)
    {
      define_variable(stmt.output);
      use_variables(stmt.cowns);
      use_variables(stmt.captures);
    }

    void visit_inner_stmt(const StaticTypeStmt& stmt)
    {
      define_variable(stmt.output);
    }

    void visit_inner_stmt(const NewStmt& stmt)
    {
      define_variable(stmt.output);
    }

    void visit_inner_stmt(const MatchBindStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.input);
    }

    void visit_inner_stmt(const ReadFieldStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.base);
    }

    void visit_inner_stmt(const WriteFieldStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.base);
      use_variable(stmt.right);
    }

    void visit_inner_stmt(const ViewStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.input);
    }

    void visit_inner_stmt(const CopyStmt& stmt)
    {
      define_variable(stmt.output);
      use_variable(stmt.input);
    }

    void visit_inner_stmt(const IntegerLiteralStmt& stmt)
    {
      define_variable(stmt.output);
    }

    void visit_inner_stmt(const StringLiteralStmt& stmt)
    {
      define_variable(stmt.output);
    }

    void visit_inner_stmt(const UnitStmt& stmt)
    {
      define_variable(stmt.output);
    }

    void visit_inner_stmt(const EndScopeStmt& stmt)
    {
      for (Variable dead_variable : stmt.dead_variables)
      {
        result_.kill_variable(dead_variable);
      }
    }

    void visit_inner_stmt(const OverwriteStmt& stmt)
    {
      result_.kill_variable(stmt.dead_variable);
    }

    void visit_inner_term(const BranchTerminator& term)
    {
      result_.phi_inputs(term.phi_arguments);
    }
    void visit_inner_term(const MatchTerminator& term)
    {
      use_variable(term.input);
    }
    void visit_inner_term(const IfTerminator& term)
    {
      use_variable(term.input);
    }
    void visit_inner_term(const ReturnTerminator& term)
    {
      use_variable(term.input);
    }

    void use_variable(IRInput input)
    {
      result_.use_variable(input);
    }
    void use_variables(const std::vector<IRInput>& inputs)
    {
      for (IRInput input : inputs)
      {
        use_variable(input);
      }
    }
    void define_variable(Variable variable)
    {
      result_.define_variable(variable);
    }

  private:
    Result& result_;
  };

  /**
   * Overall use-def of a basic block.
   *
   * The uses only includes the ones that are upwards exposed, that is they
   * aren't preceded by a definition in the same block.
   */
  struct UseDefAnalysis
  {
    VariableSet defs;
    VariableSet uses;
    VariableSet kills;

    static UseDefAnalysis compute(const BasicBlock* bb)
    {
      UseDefAnalysis result;
      UseDefVisitor<UseDefAnalysis>(result).visit_basic_block(bb);
      return result;
    }

  private:
    friend UseDefVisitor<UseDefAnalysis>;

    // These two functions are used by UseDefVisitor.
    void kill_variable(Variable variable)
    {
      kills.insert(variable);
    }
    void use_variable(IRInput input)
    {
      kills.remove(input.variable);
      uses.insert(input.variable);
    }
    void define_variable(Variable variable)
    {
      kills.remove(variable);
      uses.remove(variable);
      defs.insert(variable);
    }
    void phi_inputs(const std::vector<Variable>& vs) {}
    void phi_outputs(const std::vector<Variable>& vs) {}
  };
}
