// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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

  /**
   * Randomly generated verona bytes :)
   */
  constexpr static const char* verona_bytes = "\xB0\x45\x46\xBE";

  constexpr static uint32_t MAJOR_VERSION = 0;
  constexpr static uint32_t MINOR_VERSION = 0;
  constexpr static uint32_t BUILD_VERSION = 0;

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
      context.print_global_diagnostic(
        std::cerr, DiagnosticKind::Error, Diagnostic::NoMainClass);
      return std::nullopt;
    }

    if (main_class->kind->value() != Entity::Class)
    {
      context.print_diagnostic(
        std::cerr,
        main_class->name.source_range.first,
        DiagnosticKind::Error,
        Diagnostic::MainNotAClass);
      context.print_line_diagnostic(std::cerr, main_class->name.source_range);
      return std::nullopt;
    }

    if (!main_class->generics->types.empty())
    {
      context.print_diagnostic(
        std::cerr,
        main_class->name.source_range.first,
        DiagnosticKind::Error,
        Diagnostic::MainClassIsGeneric);
      context.print_line_diagnostic(std::cerr, main_class->name.source_range);
      return std::nullopt;
    }

    const Method* main_method = lookup_member<Method>(main_class, "main");
    if (!main_method)
    {
      context.print_diagnostic(
        std::cerr,
        main_class->name.source_range.first,
        DiagnosticKind::Error,
        Diagnostic::NoMainMethod);
      context.print_line_diagnostic(std::cerr, main_class->name.source_range);
      return std::nullopt;
    }

    if (!is_valid_main_signature(context, *main_method->signature))
    {
      context.print_diagnostic(
        std::cerr,
        main_method->name.source_range.first,
        DiagnosticKind::Error,
        Diagnostic::InvalidMainSignature);
      context.print_line_diagnostic(std::cerr, main_method->name.source_range);
      return std::nullopt;
    }

    CodegenItem<Entity> class_item(main_class, Instantiation::empty());
    CodegenItem<Method> method_item(main_method, Instantiation::empty());
    return std::make_pair(class_item, method_item);
  }

  void write_magic_bytes(std::vector<uint8_t>& vector)
  {
    for (uint8_t i = 0; i < 4; i++)
    {
      vector.push_back(verona_bytes[i]);
    }

    auto* major_version = reinterpret_cast<uint8_t*>(MAJOR_VERSION);
    auto* minor_version = reinterpret_cast<uint8_t*>(MINOR_VERSION);
    auto* build_version = reinterpret_cast<uint8_t*>(BUILD_VERSION);

    for (size_t i = 0; i < 4; i++)
    {
        vector.push_back(major_version[i]);
    }

    for (size_t i = 0; i < 4; i++)
    {
        vector.push_back(minor_version[i]);
    }
    for (size_t i = 0; i < 4; i++)
    {
        vector.push_back(build_version[i]);
    }
  }

  std::vector<uint8_t> codegen(
    Context& context, const Program& program, const AnalysisResults& analysis)
  {
    auto entry = find_entry(context, program);
    if (!entry)
      return {};

    std::vector<uint8_t> code;

    // TODO: write bytes

    write_magic_bytes(code);

    Generator gen(code);

    Reachability reachability = compute_reachability(
      context, program, gen, entry->first, entry->second, analysis);
    SelectorTable selectors = SelectorTable::build(reachability);

    emit_program_header(program, reachability, selectors, gen, entry->first);
    emit_functions(context, analysis, reachability, selectors, gen);

    gen.finish();

    return code;
  }
}
