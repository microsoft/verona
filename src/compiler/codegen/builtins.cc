// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/builtins.h"

#include "compiler/codegen/generator.h"

namespace verona::compiler
{
  using bytecode::Opcode;

  /* static */
  void BuiltinGenerator::generate(
    Context& context, Generator& gen, const CodegenItem<Method>& method)
  {
    FunctionABI abi(*method.definition->signature);
    BuiltinGenerator v(context, gen, abi);
    v.generate_header(method.instantiated_path());
    v.generate_builtin(
      method.definition->parent->name, method.definition->name);
    v.finish();
  }

  void BuiltinGenerator::generate_builtin(
    std::string_view entity, std::string_view method)
  {
    if (entity == "Builtin")
    {
      if (method.rfind("print", 0) == 0)
        return builtin_print();
      else if (method == "freeze")
        return builtin_freeze();
      else if (method == "trace")
        return builtin_trace_region();
    }
    else if (entity == "U64")
    {
      if (method == "add")
        return builtin_binop(bytecode::BinaryOperator::Add);
      else if (method == "sub")
        return builtin_binop(bytecode::BinaryOperator::Sub);
      else if (method == "mul")
        return builtin_binop(bytecode::BinaryOperator::Mul);
      else if (method == "div")
        return builtin_binop(bytecode::BinaryOperator::Div);
      else if (method == "mod")
        return builtin_binop(bytecode::BinaryOperator::Mod);
      else if (method == "shl")
        return builtin_binop(bytecode::BinaryOperator::Shl);
      else if (method == "shr")
        return builtin_binop(bytecode::BinaryOperator::Shr);
      else if (method == "lt")
        return builtin_binop(bytecode::BinaryOperator::Lt);
      else if (method == "gt")
        return builtin_binop(bytecode::BinaryOperator::Gt);
      else if (method == "le")
        return builtin_binop(bytecode::BinaryOperator::Le);
      else if (method == "ge")
        return builtin_binop(bytecode::BinaryOperator::Ge);
      else if (method == "eq")
        return builtin_binop(bytecode::BinaryOperator::Eq);
      else if (method == "ne")
        return builtin_binop(bytecode::BinaryOperator::Ne);
      else if (method == "and")
        return builtin_binop(bytecode::BinaryOperator::And);
      else if (method == "or")
        return builtin_binop(bytecode::BinaryOperator::Or);
    }
    else if (entity == "cown")
    {
      if (method == "create")
        return builtin_cown_create();
      else if (method == "_create_sleeping")
        return builtin_cown_create_sleeping();
      else if (method == "_fulfill_sleeping")
        return builtin_cown_fulfill_sleeping();
    }
    fmt::print(stderr, "Invalid builtin {}.{}\n", entity, method);
    abort();
  }

  void BuiltinGenerator::builtin_print()
  {
    // The method can generate a print method with any arity
    // It needs at least 2 arguments, for the receiver and the format string.
    assert(abi_.arguments >= 2);
    assert(abi_.returns == 1);

    std::vector<Register> args;
    for (uint8_t i = 0; i < abi_.arguments - 2; i++)
    {
      args.push_back(Register(2 + i));
    }

    emit<Opcode::Print>(Register(1), args);

    // Re-use the args vector for the clear OP, but this time we want to include
    // the first two parameters.
    args.push_back(Register(1));
    args.push_back(Register(0));

    emit<Opcode::ClearList>(args);
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_freeze()
  {
    assert(abi_.arguments == 2);
    assert(abi_.returns == 1);

    emit<Opcode::Freeze>(Register(0), Register(1));
    emit<Opcode::Clear>(Register(1));
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_trace_region()
  {
    assert(abi_.arguments == 2);
    assert(abi_.returns == 1);

    emit<Opcode::TraceRegion>(Register(1));
    emit<Opcode::ClearList>(bytecode::RegisterSpan{Register(0), Register(1)});
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_binop(bytecode::BinaryOperator op)
  {
    assert(abi_.arguments == 2);
    assert(abi_.returns == 1);

    emit<Opcode::BinOp>(Register(0), op, Register(0), Register(1));
    emit<Opcode::Clear>(Register(1));
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_cown_create()
  {
    assert(abi_.arguments == 2);
    assert(abi_.returns == 1);

    // This is a static method, therefore register 0 contains the descriptor for
    // cown[T]. We use that to initialize the cown.
    emit<Opcode::NewCown>(Register(0), Register(0), Register(1));
    emit<Opcode::Clear>(Register(1));
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_cown_create_sleeping()
  {
    assert(abi_.arguments == 1);
    assert(abi_.returns == 1);

    // This is a static method, therefore register 0 contains the descriptor for
    // cown[T]. We use that to initialize the cown.
    emit<Opcode::NewSleepingCown>(Register(0), Register(0));
    emit<Opcode::Return>();
  }

  void BuiltinGenerator::builtin_cown_fulfill_sleeping()
  {
    assert(abi_.arguments == 2);
    assert(abi_.returns == 1);

    emit<Opcode::FulfillSleepingCown>(Register(0), Register(1));
    emit<Opcode::ClearList>(bytecode::RegisterSpan{Register(0), Register(1)});
    emit<Opcode::Return>();
  }
}
