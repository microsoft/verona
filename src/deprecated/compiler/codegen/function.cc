// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/function.h"

#include "compiler/codegen/builtins.h"
#include "compiler/codegen/ir.h"
#include "compiler/printing.h"
#include "compiler/typecheck/typecheck.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  using bytecode::DescriptorIdx;
  using bytecode::Opcode;

  RegisterAllocator::RegisterAllocator(const FunctionABI& abi)
  : next_register_(abi.callspace())
  {}

  Register RegisterAllocator::get()
  {
    if (next_register_ + children_call_space_ >= bytecode::REGISTER_COUNT)
      throw std::logic_error("Ran out of registers");

    return Register(truncate<uint8_t>(next_register_++));
  }

  void RegisterAllocator::reserve_child_callspace(const FunctionABI& abi)
  {
    if (next_register_ + abi.callspace() >= bytecode::REGISTER_COUNT)
      throw std::logic_error("Ran out of registers");

    children_call_space_ = std::max(children_call_space_, abi.callspace());
  }

  uint8_t RegisterAllocator::frame_size() const
  {
    return truncate<uint8_t>(next_register_ + children_call_space_);
  }

  FunctionGenerator::FunctionGenerator(
    Context& context, Generator& gen, FunctionABI abi)
  : context_(context), gen_(gen), abi_(abi)
  {}

  void FunctionGenerator::generate_header(std::string_view name)
  {
    gen_.str(name);
    gen_.u8(truncate<uint8_t>(abi_.arguments));
    gen_.u8(truncate<uint8_t>(abi_.returns));
    gen_.u8(frame_size_);
    size_t header_end = gen_.current_offset() + 4;
    gen_.u32(end_label_, header_end);
    assert(gen_.current_offset() == header_end);
  }

  void FunctionGenerator::finish()
  {
    gen_.define_label(end_label_);
    gen_.define_relocatable(frame_size_, allocator_.frame_size());
  }

  void emit_function(
    Context& context,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Method>& method,
    const FnAnalysis& analysis)
  {
    FunctionABI abi(*method.definition->signature);

    std::vector<Label> closure_labels;
    MethodIR& mir = *analysis.ir;
    for (size_t i = 0; i < mir.function_irs.size(); i++)
      closure_labels.push_back(gen.create_label());

    for (size_t i = 0; i < mir.function_irs.size(); i++)
    {
      FunctionIR& ir = *mir.function_irs[i];
      if (i != 0)
        abi = FunctionABI::create_closure_abi(ir.parameters.size());

      IRGenerator v(
        context,
        reachability,
        selectors,
        gen,
        abi,
        method,
        *analysis.typecheck,
        *analysis.liveness,
        closure_labels);

      std::string name = method.instantiated_path();
      if (i != 0)
        name += ".$c." + std::to_string(i);

      gen.define_label(closure_labels[i]);

      v.generate_header(name);
      v.generate_body(ir);
      v.finish();
    }
  }

  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen)
  {
    for (const auto& [entity, entity_info] : reachability.entities)
    {
      for (const auto& [method, method_info] : entity_info.methods)
      {
        if (!method_info.label.has_value())
          continue;

        gen.define_label(method_info.label.value());
        if (method.definition->kind() == Method::Builtin)
        {
          BuiltinGenerator::generate(context, gen, method);
        }
        else
        {
          const FnAnalysis& fn_analysis =
            analysis.functions.at(method.definition);
          emit_function(
            context, reachability, selectors, gen, method, fn_analysis);
        }
      }
    }
  }
}
