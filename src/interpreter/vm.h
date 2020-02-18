// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "interpreter/code.h"

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace verona::interpreter
{
  using bytecode::Register;

  class VM
  {
  public:
    VM(const Code& code, bool verbose)
    : code_(code), verbose_(verbose), alloc_(rt::ThreadAlloc::get())
    {}

    static inline thread_local VM* local_vm = nullptr;

    static void dealloc_vm()
    {
      delete local_vm;
    }

    static void init_vm(const Code* code, bool verbose)
    {
      static thread_local snmalloc::OnDestruct<dealloc_vm> foo;
      local_vm = new VM(*code, verbose);
    }

    /**
     * Run the VM from the given address.
     *
     * Puts args on the stack.
     *
     * Keeps fetching and executing instructions until the VM halts.
     */
    void run(std::vector<Value> args, size_t cown_count, size_t start);

    /**
     * Run finaliser for this VM object.
     *
     * This creates a new frame in the thread local VMs state. Existing running
     * frames will be restored after the finaliser completes.
     *
     * This assumes that `object` does indeed have a finaliser, found in its
     * descriptor's finaliser_ip field.
     **/
    static void execute_finaliser(VMObject* object);

  private:
    void opcode_binop(
      Register dst,
      bytecode::BinaryOperator op,
      const Value& left,
      const Value& right);
    void opcode_call(SelectorIdx selector, uint8_t callspace);
    void call(size_t addr, uint8_t callspace);
    void opcode_clear(Register dst);
    void opcode_copy(Register dst, Value src);
    void opcode_fulfill_sleeping_cown(const Value& cown, Value result);
    void opcode_freeze(Register dst, Value src);
    void opcode_int64(Register dst, uint64_t imm);
    void opcode_jump(int16_t offset);
    void opcode_jump_if(const Value& src, int16_t offset);
    void opcode_load(Register dst, const Value& base, SelectorIdx selector);
    void opcode_load_descriptor(Register dst, DescriptorIdx desc_idx);
    void opcode_match(
      Register dst, const Value& src, const VMDescriptor* descriptor);
    void opcode_move(Register dst, Register src);
    void opcode_mut_view(Register dst, const Value& src);
    void opcode_new(
      Register dst, const Value& parent, const VMDescriptor* descriptor);
    void opcode_new_region(Register dst, const VMDescriptor* descriptor);
    void
    opcode_new_cown(Register dst, const VMDescriptor* descriptor, Value src);
    void opcode_new_sleeping_cown(Register dst, const VMDescriptor* descriptor);
    void opcode_print(const Value& src, uint8_t argc);
    void opcode_return();
    void opcode_store(
      Register dst, const Value& base, SelectorIdx selector, Value src);
    void opcode_string(Register dst, std::string_view imm);
    void opcode_trace_region(const Value& region);
    void
    opcode_when(CodePtr selector, uint8_t cown_count, uint8_t capture_count);
    void opcode_unreachable();

    /**
     * Switches on the opcode value and invokes the appropriate handler.
     */
    void dispatch_opcode(Opcode op);

    /**
     * Executes the VMs IP until the it returns from outer most stack frame.
     **/
    void dispatch_loop();

    /**
     * Wrapper around opcode handlers. Takes care of parsing and tracing the
     * operands.
     *
     * Fn is the actual handler implementation, which will be called with the
     * operands as arguments. It should be a member function pointer of the VM
     * class.
     */
    template<Opcode opcode, auto Fn>
    void execute_opcode(size_t& ip);

    void grow_stack(size_t size);

    /**
     * Read the value of a register, relative to the current frame.
     *
     * Aborts the VM if the register is out of bounds.
     */
    Value& read(Register reg);
    const Value& read(Register reg) const;

    /**
     * Write a value to a register, relative to the current frame.
     *
     * Aborts the VM if the register is out of bounds.
     */
    void write(Register reg, Value value);

    const VMDescriptor* find_dispatch_descriptor(Register receiver) const;

    template<typename... Args>
    void trace(std::string_view fmt, Args&&... args) const
    {
      if (verbose_)
      {
        fmt::print(std::cerr, "[{:4x}]: {:<{}}", start_ip_, "", indent_);
        fmt::print(std::cerr, fmt, std::forward<Args>(args)...);
        fmt::print(std::cerr, "\n");
      }
    }

    template<typename... Args>
    [[noreturn]] void fatal(std::string_view fmt, Args&&... args) const
    {
      fmt::print(std::cerr, "[{:4x}]: {:<{}}FATAL: ", start_ip_, "", indent_);
      fmt::print(std::cerr, fmt, std::forward<Args>(args)...);
      fmt::print(std::cerr, "\n");
      abort();
    }

    void check_type(const Value& value, Value::Tag expected);
    void check_type(const Value& value, std::vector<Value::Tag> expected);

    const Code& code_;
    rt::Alloc* alloc_;
    bool verbose_;

    /**
     * Instruction Pointer
     */
    size_t ip_;

    /**
     * Address of the currently executing instruction.
     *
     * After an instruction is parsed, ip_ points to the start of the next
     * instruction. start_ip_ is used for tracing, and to resolve relative
     * offsets.
     */
    size_t start_ip_;

    /**
     * Flag to halt VM execution.
     *
     * The run() method will return once an opcode handler sets this field to
     * true.
     */
    bool halt_;

    /**
     * Value stack.
     *
     * Each execution frame has a view into this stack. Register access is done
     * within this view.
     */
    std::vector<Value> stack_;

    struct Frame
    {
      /**
       * Address to which execution returns when exiting this frame.
       *
       * This is unused in the lowest frame, as exiting that frame halts the VM.
       */
      size_t return_address;

      /**
       * Base offset into the value stack.
       *
       * All register accesses are relative the the current frame's base
       */
      size_t base;

      /**
       * Number of registers accessible from this frame.
       */
      uint8_t locals;

      /**
       * Number of argument and return registers for the current frame.
       */
      uint8_t argc;
      uint8_t retc;

      static Frame initial()
      {
        return Frame{0, 0, 0, 0, 0};
      }
    };

    /**
     * Current frame.
     */
    Frame frame_;

    /**
     * Call stack.
     *
     * On CALL, the current frame is pushed on here, and on RETURN the top of
     * the stack is popped and becomes the current frame.
     */
    std::vector<Frame> cfstack_;

    /**
     * Ident level for tracing
     **/
    size_t indent_ = 0;

    /**
     * Helper type used to convert a single operand.
     *
     * The class should be instantiated using the opcode handler method's
     * argument type. Every specialization of this class provides a method with
     * the following signature:
     *
     *     static T convert(VM*, U operand)
     *
     * Where U is the type of the argument as described in the opcode's spec.
     *
     * For example there is a convert_operand<Value> specialization,
     * which provides a `static Value convert(VM*, Register)` method. This
     * allows loading of registers to be omitted from handler's bodies, by
     * making them accept a Value argument directly.
     */
    template<typename T>
    struct convert_operand;

    /**
     * Helper type extending convert_operand to a list of operands.
     *
     * This class should be instantiated using the type of the member function
     * pointer to an opcode handler, such as `void (VM::*)(Register, Value)`.
     *
     * It provides the following method:
     *
     *     template<Ts...>
     *     static tuple<...> convert(VM*, tuple<Ts...> operands)
     *
     * Where the argument tuple matches the opcode's spec, and the return tuple
     * matches the opcode handler's argument types.
     */
    template<typename Fn>
    struct convert_operand_list;
    template<typename... Args>
    struct convert_operand_list<void (VM::*)(Args...)>
    {
      template<typename... Ts>
      static std::tuple<Args...> convert(VM* vm, std::tuple<Ts...> operands)
      {
        return std::apply(
          [&](auto... operands) -> std::tuple<Args...> {
            return {convert_operand<Args>::convert(vm, operands)...};
          },
          operands);
      }
    };
  };

  /**
   * Blanket identity conversion on operands, for any type.
   */
  template<typename T>
  struct VM::convert_operand
  {
    static T convert(VM* vm, T value)
    {
      return value;
    }
  };

  /**
   * Operand conversion from a Register to a Value, which might consume the
   * underlying register. The opcode handler receives ownership of the Value and
   * must use it or clear it before exiting.
   */
  template<>
  struct VM::convert_operand<Value>
  {
    static Value convert(VM* vm, Register reg)
    {
      return vm->read(reg).maybe_consume();
    }
  };

  /**
   * Operand conversion from a Register to a borrowed Value.
   */
  template<>
  struct VM::convert_operand<const Value&>
  {
    static const Value& convert(VM* vm, Register reg)
    {
      return vm->read(reg);
    }
  };

  /**
   * Operand conversion from a Register to a VMDescriptor.
   *
   * This conversion loads the register, assuming it has tag DESCRIPTOR, and
   * returns its contents.
   */
  template<>
  struct VM::convert_operand<const VMDescriptor*>
  {
    static const VMDescriptor* convert(VM* vm, Register reg)
    {
      const Value& value = vm->read(reg);
      vm->check_type(value, Value::Tag::DESCRIPTOR);
      return value->descriptor;
    }
  };

  /**
   * This represent the closure for all when clauses in the runtime
   */
  class ExecuteMessage : public rt::VAction<ExecuteMessage>
  {
    size_t start;
    std::vector<Value> args;
    size_t cown_count;

  public:
    ExecuteMessage(size_t start, std::vector<Value> args, size_t cown_count)
    : start(start), args(std::move(args)), cown_count(cown_count)
    {}

    // Main runtime entry for a closure.
    void f()
    {
      VM::local_vm->run(std::move(args), cown_count, start);
    }
  };
}
