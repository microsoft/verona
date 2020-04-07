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
    Value
    opcode_binop(bytecode::BinaryOperator op, uint64_t left, uint64_t right);
    void opcode_call(SelectorIdx selector, uint8_t callspace);
    void call(size_t addr, uint8_t callspace);
    Value opcode_clear();
    Value opcode_copy(Value src);
    void opcode_error(std::string_view reason);
    void opcode_fulfill_sleeping_cown(const Value& cown, Value result);
    Value opcode_freeze(Value src);
    Value opcode_int64(uint64_t imm);
    void opcode_jump(int16_t offset);
    void opcode_jump_if(uint64_t condition, int16_t offset);
    Value opcode_load(const Value& base, SelectorIdx selector);
    Value opcode_load_descriptor(DescriptorIdx desc_idx);
    Value opcode_match(const Value& src, const VMDescriptor* descriptor);
    Value opcode_move(Register src);
    Value opcode_mut_view(const Value& src);
    Value
    opcode_new_object(const Value& parent, const VMDescriptor* descriptor);
    Value opcode_new_region(const VMDescriptor* descriptor);
    Value opcode_new_cown(const VMDescriptor* descriptor, Value src);
    Value opcode_new_sleeping_cown(const VMDescriptor* descriptor);
    void opcode_print(std::string_view fmt, uint8_t argc);
    void opcode_return();
    Value opcode_store(const Value& base, SelectorIdx selector, Value src);
    Value opcode_string(std::string_view imm);
    void opcode_trace_region(const Value& region);
    void
    opcode_when(CodePtr selector, uint8_t cown_count, uint8_t capture_count);
    void opcode_unreachable();

    Value opcode_pointer_allocate(uint64_t size);
    void opcode_pointer_free(const Value& parent, uint64_t size);
    Value
    opcode_pointer_get(const Value& parent, const Value& ptr, uint64_t index);
    void opcode_pointer_set(
      const Value& parent, const Value& ptr, uint64_t index, Value value);
    Value opcode_pointer_swap(
      const Value& parent, const Value& ptr, uint64_t index, Value value);
    void opcode_pointer_move(
      const Value& parent, const Value& src, const Value& dst, uint64_t size);

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

    template<typename T>
    friend struct convert_operand;
    template<typename T>
    friend struct execute_handler;
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
