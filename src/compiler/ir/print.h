// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/liveness.h"
#include "compiler/ir/ir.h"
#include "compiler/typecheck/typecheck.h"

namespace verona::compiler
{
  /**
   * Utility class to print a function's IR.
   *
   * Additional information, such as inferred types or liveness can be printing
   * as well, by calling the `with_XXX` methods before calling `print`.
   */
  class IRPrinter
  {
  public:
    explicit IRPrinter(std::ostream& out) : out_(out) {}

    /**
     * Print typechecking information with the IR.
     */
    IRPrinter& with_types(const TypecheckResults& types)
    {
      types_ = &types;
      return *this;
    }

    /**
     * Print liveness analysis information with the IR.
     */
    IRPrinter& with_liveness(const LivenessAnalysis& liveness)
    {
      liveness_ = &liveness;
      return *this;
    }

    void print(
      const std::string& title, const Method& method, const MethodIR& ir) const;

    void print_header(const std::string& title, const Method& method) const;
    void print_entry(const FunctionIR& ir) const;
    void print_body(const FunctionIR& ir) const;

    void print_basic_block(const BasicBlock* bb) const;
    void print_basic_block_header(const BasicBlock* bb) const;
    void print_basic_block_body(const BasicBlock* bb) const;

    void print_statement(
      const TypeAssignment* assignment, const Statement& stmt) const;
    void print_terminator(const Terminator& terminator) const;

  private:
    void print_liveness(const Liveness& live) const;

    void print_inner_statement(
      const TypeAssignment* assignment, const NewStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const CallStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const WhenStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const StaticTypeStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const MatchBindStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const ReadFieldStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const WriteFieldStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const ViewStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const CopyStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const IntegerLiteralStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const StringLiteralStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const UnitStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const EndScopeStmt& stmt) const;
    void print_inner_statement(
      const TypeAssignment* assignment, const OverwriteStmt& stmt) const;

    void print_inner_terminator(const BranchTerminator& terminator) const;
    void print_inner_terminator(const ReturnTerminator& terminator) const;
    void print_inner_terminator(const IfTerminator& terminator) const;
    void print_inner_terminator(const MatchTerminator& terminator) const;

    std::string type_of(const TypeAssignment* assignment, Variable var) const;
    std::string type_arguments(TypeArgumentsId arguments_id) const;

    std::ostream& out_;

    const TypecheckResults* types_ = nullptr;
    const LivenessAnalysis* liveness_ = nullptr;
  };
}
