// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/analysis.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/codegen/generator.h"

namespace verona::compiler
{
  using bytecode::Register;

  /**
   * Generate code for a function.
   *
   * This chooses between BuiltinGenerator and IRGenerator to generate the
   * method, depending on its kind.
   */
  void emit_function(
    Context& context,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Method>& method,
    const FnAnalysis& analysis);

  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen);

  struct FunctionABI
  {
    explicit FunctionABI(const FnSignature& sig)
    : arguments(1 + sig.parameters.size()), returns(1)
    {}

    explicit FunctionABI(const CallStmt& stmt)
    : arguments(1 + stmt.arguments.size()), returns(1)
    {}

    // Adds one to arguments for unused receiver
    // TODO-Better-Static-codegen
    // No output for now TODO-PROMISE
    explicit FunctionABI(const WhenStmt& stmt)
    : arguments(stmt.cowns.size() + stmt.captures.size() + 1), returns(1)
    {}

    /**
     * Number of arguments this function has.
     *
     * Note that the ABI always has a receiver argument, even if the function is
     * a static method. There may therefore be 1 register argument more than the
     * function has parameters.
     */
    size_t arguments;

    /**
     * Number of return values this function has, currently always 1.
     */
    size_t returns;

    /**
     * Get the space needed to pass arguments and return values.
     *
     * This is the size of the overlap between the caller's and the callee's
     * frames
     */
    size_t callspace() const
    {
      return std::max(arguments, returns);
    }

    static FunctionABI create_closure_abi(size_t count)
    {
      return FunctionABI(count);
    }

  private:
    // Adds one to arguments for unused receiver
    // TODO-Better-Static-codegen
    explicit FunctionABI(size_t count) : arguments(count + 1), returns(1) {}
  };

  class RegisterAllocator
  {
  public:
    RegisterAllocator(const FunctionABI& abi);

    /**
     * Allocate a new register.
     */
    Register get();

    /**
     * Reserve space at the top of the frame used to pass arguments during
     * function calls.
     *
     * The same space is used for all calls in the current function. Registers
     * returned by the `get` function will never overlap with this area.
     */
    void reserve_child_callspace(const FunctionABI& abi);

    /**
     * Get the total number of registers needed by the function.
     */
    uint8_t frame_size() const;

  private:
    size_t next_register_;
    size_t children_call_space_ = 0;
  };

  /**
   * Utility class for generating function code.
   *
   * This class does not actually generate the function's body, but handles all
   * the extra work, such as generating the right header, register allocation
   * and emitting child relative registers.
   *
   * The actual body generation depends on the kind of function being
   * generated (i.e. builtin vs IR).
   */
  class FunctionGenerator
  {
  public:
    FunctionGenerator(Context& context, Generator& gen, FunctionABI abi);

    /**
     * Emit the function's header.
     */
    void generate_header(std::string_view name);

    /**
     * Finish generating the function
     *
     * This must only be called once, after the function's body was generated.
     */
    void finish();

    CalleeRegister callee_register(const FunctionABI& abi, uint8_t index)
    {
      return CalleeRegister(abi.callspace(), frame_size_, Register(index));
    }

    template<bytecode::Opcode Op, typename... Ts>
    void emit(Ts&&... ts)
    {
      gen_.emit<Op>(std::forward<Ts>(ts)...);
    }

    Label create_label()
    {
      return gen_.create_label();
    }

    void define_label(Label label)
    {
      gen_.define_label(label);
    }

  protected:
    Context& context_;
    FunctionABI abi_;

    RegisterAllocator allocator_ = RegisterAllocator(abi_);

  private:
    Generator& gen_;

    /**
     * Total number of registers used by the function.
     * This is used in the function header, and when accessing child-relative
     * registers.
     */
    Generator::Relocatable frame_size_ = gen_.create_relocatable();

    /**
     * Address of the end of the function. Used to compute the total function
     * size in the function header.
     */
    Label end_label_ = gen_.create_label();
  };
}
