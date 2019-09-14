// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "compiler/codegen/builtins.h"

#include "compiler/codegen/generator.h"

namespace verona::compiler
{
  using bytecode::Opcode;

  void BuiltinGenerator::generate_builtin(std::string_view name)
  {
    if (name.rfind("print", 0) == 0)
      builtin_print();
    else if (name == "u64_add")
      builtin_binop(bytecode::BinaryOperator::Add);
    else if (name == "u64_sub")
      builtin_binop(bytecode::BinaryOperator::Sub);
    else if (name == "u64_lt")
      builtin_binop(bytecode::BinaryOperator::Lt);
    else if (name == "u64_gt")
      builtin_binop(bytecode::BinaryOperator::Gt);
    else if (name == "u64_le")
      builtin_binop(bytecode::BinaryOperator::Le);
    else if (name == "u64_ge")
      builtin_binop(bytecode::BinaryOperator::Ge);
    else if (name == "u64_eq")
      builtin_binop(bytecode::BinaryOperator::Eq);
    else if (name == "u64_ne")
      builtin_binop(bytecode::BinaryOperator::Ne);
    else if (name == "u64_and")
      builtin_binop(bytecode::BinaryOperator::And);
    else if (name == "u64_or")
      builtin_binop(bytecode::BinaryOperator::Or);
    else if (name == "create_sleeping_cown")
      builtin_create_sleeping_cown();
    else if (name == "fulfill_sleeping_cown")
      builtin_fulfill_sleeping_cown();
    else
      throw std::logic_error("Invalid builtin");
  }

  void BuiltinGenerator::builtin_print()
  {
    // The method can generate a print method with any arity
    // It needs at least 2 arguments, for the receiver and the format string.
    assert(abi_.arguments >= 2);
    assert(abi_.returns == 1);

    size_t value_count = abi_.arguments - 2;
    uint8_t value_count_trunc = truncate<uint8_t>(value_count);
    gen_.opcode(Opcode::Print);
    gen_.reg(Register(1));
    gen_.u8(value_count_trunc);
    for (uint8_t i = 0; i < value_count_trunc; i++)
    {
      gen_.reg(Register(2 + i));
    }

    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(1));
    for (uint8_t i = 0; i < value_count_trunc; i++)
    {
      gen_.opcode(Opcode::Clear);
      gen_.reg(Register(2 + i));
    }

    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(0));
    gen_.opcode(Opcode::Return);
  }

  void BuiltinGenerator::builtin_create_sleeping_cown()
  {
    assert(abi_.arguments == 1);
    assert(abi_.returns == 1);

    gen_.opcode(Opcode::NewSleepingCown);
    gen_.reg(Register(0));
    gen_.opcode(Opcode::Return);
  }

  void BuiltinGenerator::builtin_fulfill_sleeping_cown()
  {
    assert(abi_.arguments == 3);
    assert(abi_.returns == 1);

    gen_.opcode(Opcode::FulfillSleepingCown);
    gen_.reg(Register(1));
    gen_.reg(Register(2));
    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(0));
    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(1));
    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(2));
    gen_.opcode(Opcode::Return);
  }

  void BuiltinGenerator::builtin_binop(bytecode::BinaryOperator op)
  {
    assert(abi_.arguments == 3);
    assert(abi_.returns == 1);

    gen_.opcode(Opcode::BinOp);
    gen_.reg(Register(0));
    gen_.u8(static_cast<uint8_t>(op));
    gen_.reg(Register(1));
    gen_.reg(Register(2));
    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(1));
    gen_.opcode(Opcode::Clear);
    gen_.reg(Register(2));
    gen_.opcode(Opcode::Return);
  }
}
