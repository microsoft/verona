// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/value_list.h"
#include "interpreter/vm.h"

namespace verona::interpreter
{
  /**
   * Helper type used to convert operands.
   *
   * The opcode spec defines the types of operands as they are "on the wire".
   * However the opcode handlers, defined as methods in VM, may use higher level
   * types. For example, rather than receiving a `Register` argument, which is
   * nothing more than an 8-bit index into the frame, opcodes may use `const
   * Value&`, which is a reference to the actual value held in the frame.
   *
   * `convert_operand` bridges the gap between the wire type and the handler
   * argument, by describing how to convert from one to the other.
   *
   * Given an argument type `T` and a wire value `operand`, the following call
   * performs the conversion:
   *
   *   convert_operand<T>::convert(vm, operand);
   *
   * The conversions are implemented by specializing the `convert_operand`
   * struct. Specializations are picked using the argument type. Each
   * specialization may provide one or more `convert` methods, the appropriate
   * one will be selected using the usual C++ overload resolution, based on the
   * wire type of the operand.
   */
  template<typename T>
  struct convert_operand;

  /**
   * Blanket identity conversion on operands, for any type.
   */
  template<typename T>
  struct convert_operand
  {
    static T convert(VM* vm, T value)
    {
      return value;
    }
  };

  template<>
  struct convert_operand<Value>
  {
    /**
     * Operand conversion from a Register to a Value, which might consume the
     * underlying register. The opcode handler receives ownership of the Value
     * and must use it or clear it before exiting.
     */
    static Value convert(VM* vm, Register reg)
    {
      return vm->read(reg).maybe_consume();
    }
  };

  template<>
  struct convert_operand<const Value&>
  {
    /**
     * Operand conversion from a Register to a borrowed Value.
     */
    static const Value& convert(VM* vm, Register reg)
    {
      return vm->read(reg);
    }
  };

  template<>
  struct convert_operand<const VMDescriptor*>
  {
    /**
     * Operand conversion from a Register to a VMDescriptor.
     *
     * This conversion loads the register, assuming it has tag DESCRIPTOR, and
     * returns its contents.
     */
    static const VMDescriptor* convert(VM* vm, Register reg)
    {
      const Value& value = vm->read(reg);
      vm->check_type(value, Value::Tag::DESCRIPTOR);
      return value->descriptor;
    }
  };

  template<>
  struct convert_operand<uint64_t>
  {
    /**
     * Identity conversion, used when the operand in the bytecode is already a
     * uint64_t.
     */
    static uint64_t convert(VM* vm, uint64_t value)
    {
      return value;
    }

    /**
     * Operand conversion from a Register to a uint64_t.
     *
     * This conversion loads the register, assuming it has tag U64, and
     * returns its contents.
     */
    static uint64_t convert(VM* vm, Register reg)
    {
      const Value& value = vm->read(reg);
      vm->check_type(value, Value::U64);
      return value->u64;
    }
  };

  template<>
  struct convert_operand<std::string_view>
  {
    /**
     * Identity conversion, used when the operand in the bytecode is already a
     * string literal.
     */
    static std::string_view convert(VM* vm, std::string_view value)
    {
      return value;
    }

    /**
     * Operand conversion from a Register to a std::string_view.
     */
    static std::string_view convert(VM* vm, Register reg)
    {
      const Value& value = vm->read(reg);
      vm->check_type(value, Value::STRING);
      return value->string();
    }
  };

  template<bool IsConst>
  struct convert_operand<BaseValueList<IsConst>>
  {
    static BaseValueList<IsConst> convert(VM* vm, RegisterSpan regs)
    {
      return BaseValueList<IsConst>(vm, regs);
    }
  };

  /**
   * Helper type used to convert operands and execute an opcode handler.
   * It is used to "pattern match" on the signature of the handler and
   * instantiate the right operand conversion.
   *
   * The type is specialized to provide different behaviours based on the return
   * type of the opcode handler. If it is `Value`, then the first operand is
   * assumed to be a register index, in which the return value is stored.
   */
  template<typename Fn>
  struct execute_handler;

  template<typename... Args>
  struct execute_handler<void (VM::*)(Args...)>
  {
    template<void (VM::*Fn)(Args...), typename... Ts>
    static void execute(VM* vm, Ts... operands)
    {
      (vm->*Fn)(convert_operand<Args>::convert(vm, operands)...);
    }
  };

  template<typename... Args>
  struct execute_handler<Value (VM::*)(Args...)>
  {
    template<Value (VM::*Fn)(Args...), typename... Ts>
    static void execute(VM* vm, Register dst, Ts... operands)
    {
      Value result = (vm->*Fn)(convert_operand<Args>::convert(vm, operands)...);
      vm->write(dst, std::move(result));
    }
  };
}
