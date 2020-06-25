// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/print.h"

#include "compiler/format.h"
#include "compiler/zip.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  void IRPrinter::print(
    const std::string& title, const Method& method, const MethodIR& mir) const
  {
    auto closure_id_counter = 0;
    for (auto& ir : mir.function_irs)
    {
      if (closure_id_counter == 0)
      {
        print_header(title, method);
        print_body(*ir);
      }
      else
      {
        fmt::print(out_, "{} for closure.{}:\n", title, closure_id_counter);
        print_body(*ir);
      }
      closure_id_counter++;
    }
  }

  void
  IRPrinter::print_header(const std::string& title, const Method& method) const
  {
    fmt::print(out_, "{} for {}:\n", title, method.path());
  }

  void IRPrinter::print_entry(const FunctionIR& ir) const
  {
    const TypeAssignment* assignment = nullptr;
    if (types_)
      assignment = &types_->types.at(ir.entry);

    if (ir.receiver)
    {
      fmt::print(
        out_,
        "Receiver\n  {}{}\n",
        *ir.receiver,
        type_of(assignment, *ir.receiver));
    }

    if (ir.parameters.size() > 0)
    {
      fmt::print(out_, "Parameters:\n");
    }

    for (const auto& var : ir.parameters)
    {
      fmt::print(out_, "  {}{}\n", var, type_of(assignment, var));
    }

    fmt::print(out_, "\n");
  }

  void IRPrinter::print_body(const FunctionIR& ir) const
  {
    print_entry(ir);

    IRTraversal traversal(ir);
    while (const BasicBlock* bb = traversal.next())
    {
      print_basic_block(bb);
    }
  }

  void IRPrinter::print_basic_block(const BasicBlock* bb) const
  {
    print_basic_block_header(bb);
    print_basic_block_body(bb);
    fmt::print(out_, "\n");
  }

  void IRPrinter::print_basic_block_header(const BasicBlock* bb) const
  {
    const TypeAssignment* assignment = nullptr;
    if (types_)
      assignment = &types_->types.at(bb);

    auto format_phi = [&](Variable phi) {
      return fmt::format("{}{}", phi, type_of(assignment, phi));
    };

    fmt::print(
      out_,
      "  Basic block {}{}:\n",
      *bb,
      format::optional(
        format::parens(format::comma_sep(bb->phi_nodes, format_phi))));
  }

  void IRPrinter::print_basic_block_body(const BasicBlock* bb) const
  {
    const TypeAssignment* assignment = nullptr;
    if (types_)
      assignment = &types_->types.at(bb);

    if (liveness_)
    {
      auto stmt_liveness = liveness_->statements_in(bb);
      for (const auto& [stmt, live] : safe_zip(bb->statements, stmt_liveness))
      {
        print_liveness(live);
        print_statement(assignment, stmt);
      }

      print_liveness(liveness_->terminator_in(bb));
      print_terminator(*bb->terminator);
    }
    else
    {
      for (const auto& stmt : bb->statements)
      {
        print_statement(assignment, stmt);
      }
      print_terminator(*bb->terminator);
    }
  }

  void IRPrinter::print_liveness(const Liveness& liveness) const
  {
    fmt::print(
      out_,
      "    live({}), zombie({})\n",
      format::comma_sep(format::sorted(liveness.live_variables)),
      format::comma_sep(format::sorted(liveness.zombie_variables)));
  }

  void IRPrinter::print_statement(
    const TypeAssignment* assignment, const Statement& stmt) const
  {
    std::visit(
      [&](const auto& inner) {
        fmt::print(out_, "    ");
        print_inner_statement(assignment, inner);
        fmt::print(out_, "\n");
      },
      stmt);
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const NewStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- new {}{}{}{}",
      stmt.output,
      stmt.definition->name,
      type_arguments(stmt.type_arguments),
      format::optional(format::prefixed(" in ", stmt.parent)),
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const CallStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- call {}.{}{}({}){}",
      stmt.output,
      stmt.receiver,
      stmt.method,
      type_arguments(stmt.type_arguments),
      format::comma_sep(stmt.arguments),
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const WhenStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- when ({}) [{}] closure.{}",
      stmt.output,
      format::comma_sep(stmt.cowns),
      format::comma_sep(stmt.captures),
      stmt.closure_index,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const StaticTypeStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- static {}{}{}",
      stmt.output,
      stmt.definition->name,
      type_arguments(stmt.type_arguments),
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const MatchBindStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- bind {}{}",
      stmt.output,
      stmt.input,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const ReadFieldStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- {}.{}{}",
      stmt.output,
      stmt.base,
      stmt.name,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const WriteFieldStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- {}.{} = {}{}",
      stmt.output,
      stmt.base,
      stmt.name,
      stmt.right,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const ViewStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- mut-view({}){}",
      stmt.output,
      stmt.input,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const CopyStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- copy {}{}",
      stmt.output,
      stmt.input,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const IntegerLiteralStmt& stmt) const
  {
    fmt::print(
      out_,
      "{} <- integer {}{}",
      stmt.output,
      stmt.value,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const StringLiteralStmt& stmt) const
  {
    // TODO: we should be escaping stmt.value
    fmt::print(
      out_,
      "{} <- string \"{}\"{}",
      stmt.output,
      stmt.value,
      type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const UnitStmt& stmt) const
  {
    fmt::print(
      out_, "{} <- unit{}", stmt.output, type_of(assignment, stmt.output));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const EndScopeStmt& stmt) const
  {
    fmt::print(
      out_,
      "end-scope({})",
      format::comma_sep(format::sorted(stmt.dead_variables)));
  }

  void IRPrinter::print_inner_statement(
    const TypeAssignment* assignment, const OverwriteStmt& stmt) const
  {
    fmt::print(out_, "overwrite({})", stmt.dead_variable);
  }

  void IRPrinter::print_terminator(const Terminator& terminator) const
  {
    std::visit(
      [&](const auto& inner) { print_inner_terminator(inner); }, terminator);
  }

  void
  IRPrinter::print_inner_terminator(const BranchTerminator& terminator) const
  {
    fmt::print(
      out_,
      "    goto {}{}\n",
      *terminator.target,
      format::optional(
        format::parens(format::comma_sep(terminator.phi_arguments))));
  }

  void
  IRPrinter::print_inner_terminator(const ReturnTerminator& terminator) const
  {
    fmt::print(out_, "    return {}\n", terminator.input);
  }

  void IRPrinter::print_inner_terminator(const IfTerminator& terminator) const
  {
    fmt::print(out_, "    if {}\n", terminator.input);
    fmt::print(out_, "     then goto {}\n", *terminator.true_target);
    fmt::print(out_, "     else goto {}\n", *terminator.false_target);
  }

  void
  IRPrinter::print_inner_terminator(const MatchTerminator& terminator) const
  {
    fmt::print(out_, "    match {}\n", terminator.input);
    for (const auto& arm : terminator.arms)
    {
      fmt::print(out_, "     case {}: goto {}\n", *arm.type, *arm.target);
    }
  }

  std::string
  IRPrinter::type_of(const TypeAssignment* assignment, Variable var) const
  {
    if (assignment)
      return fmt::format(" :: {}", *assignment->at(var));
    else
      return "";
  }

  std::string IRPrinter::type_arguments(TypeArgumentsId arguments_id) const
  {
    if (types_)
    {
      const TypeList& arguments = types_->type_arguments.at(arguments_id);
      return fmt::format("{}", format::optional_list(arguments));
    }
    else
      return "";
  }
}
