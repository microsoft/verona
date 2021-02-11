// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/codegen.h"

#include "compiler/ast.h"
#include "compiler/codegen/builtins.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/codegen/function.h"
#include "compiler/codegen/generator.h"
#include "compiler/codegen/reachability.h"
#include "compiler/instantiation.h"
#include "compiler/resolution.h"
#include "ds/helpers.h"
#include "interpreter/bytecode.h"

namespace verona::compiler
{
  using bytecode::Opcode;
  using bytecode::SelectorIdx;

  bool is_valid_main_signature(Context& context, const FnSignature& signature)
  {
    return signature.generics->types.empty() && signature.receiver == nullptr &&
      signature.types.arguments.empty() &&
      signature.types.return_type == context.mk_unit();
  }

  /**
   * Search for the program entrypoint and check it has the right signature.
   *
   * Returns nullopt and raises errors in the context if the entrypoint isn't
   * found or is invalid.
   */
  std::optional<std::pair<CodegenItem<Entity>, CodegenItem<Method>>>
  find_entry(Context& context, const Program& program)
  {
    const Entity* main_class = program.find_entity("Main");
    if (!main_class)
    {
      report(
        context, std::nullopt, DiagnosticKind::Error, Diagnostic::NoMainClass);
      return std::nullopt;
    }

    if (main_class->kind->value() != Entity::Class)
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::MainNotAClass);
      return std::nullopt;
    }

    if (!main_class->generics->types.empty())
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::MainClassIsGeneric);
      return std::nullopt;
    }

    const Method* main_method = lookup_member<Method>(main_class, "main");
    if (!main_method)
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::NoMainMethod);
      return std::nullopt;
    }

    if (!is_valid_main_signature(context, *main_method->signature))
    {
      report(
        context,
        main_method->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::InvalidMainSignature);
      return std::nullopt;
    }

    CodegenItem<Entity> class_item(main_class, Instantiation::empty());
    CodegenItem<Method> method_item(main_method, Instantiation::empty());
    return std::make_pair(class_item, method_item);
  }

  /**
   * Writes the magic numbers to the bytecode
   * @param code Generator to which the bytes should be emitted
   */
  void write_magic_number(Generator& code)
  {
    code.u32(bytecode::MAGIC_NUMBER);
  }

  std::vector<uint8_t> codegen(
    Context& context, const Program& program, const AnalysisResults& analysis)
  {
    auto entry = find_entry(context, program);
    if (!entry)
      return {};

    std::vector<uint8_t> code;

    Generator gen(code);
    write_magic_number(gen);

    Reachability reachability = compute_reachability(
      context, program, gen, entry->first, entry->second, analysis);
    SelectorTable selectors = SelectorTable::build(reachability);

    emit_program_header(program, reachability, selectors, gen, entry->first);
    emit_functions(context, analysis, reachability, selectors, gen);

    gen.finish();

    return code;
  }
}
