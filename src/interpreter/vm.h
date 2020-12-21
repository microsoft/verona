// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/code.h"

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace verona::interpreter
{
  using bytecode::AbsoluteOffset;
  using bytecode::Register;
  using bytecode::RelativeOffset;

  template<bool Const>
  class BaseValueList;

  using ValueList = BaseValueList<false>;
  using ConstValueList = BaseValueList<true>;

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
    Value opcode_clear();
    void opcode_clear_list(ValueList values);
    Value opcode_copy(Value src);
    void opcode_fulfill_sleeping_cown(const Value& cown, Value result);
    Value opcode_freeze(Value src);
    Value opcode_int64(uint64_t imm);
    void opcode_jump(RelativeOffset offset);
    void opcode_jump_if(uint64_t condition, RelativeOffset offset);
    Value opcode_load(const Value& base, SelectorIdx selector);
    Value opcode_load_descriptor(DescriptorIdx desc_idx);
    Value opcode_match_descriptor(const Value& src, const VMDescriptor* desc);
    Value opcode_match_capability(const Value& src, bytecode::Capability cap);
    Value opcode_move(Register src);
    Value opcode_mut_view(const Value& src);
    Value
    opcode_new_object(const Value& parent, const VMDescriptor* descriptor);
    Value opcode_new_region(const VMDescriptor* descriptor);
    Value opcode_new_cown(const VMDescriptor* descriptor, Value src);
    Value opcode_new_sleeping_cown(const VMDescriptor* descriptor);
    void opcode_print(std::string_view fmt, ConstValueList values);
    void opcode_protect(ConstValueList values);
    void opcode_unprotect(ConstValueList values);
    void opcode_return();
    Value opcode_store(const Value& base, SelectorIdx selector, Value src);
    Value opcode_string(std::string_view imm);
    void opcode_trace_region(const Value& region);
    void opcode_when(
      AbsoluteOffset offset, uint8_t cown_count, uint8_t capture_count);
    void opcode_unreachable();

    enum class OnReturn
    {
      Halt,
      Continue,
    };

    /**
     * Setup a new frame for execution.
     *
     * The frame is added to the control flow stack, and the register stack is
     * grown to be big enough to execute this frame.
     */
    void push_frame(size_t ip, size_t base, OnReturn on_return);

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

    /**
     * Find the descriptor used to invoke methods. Generally, this is the same
     * as the "match descriptor". However, if `value` is itself a descriptor
     * pointer, we allow methods to be called on it (ie. static method calls),
     * but matching will always fail.
     *
     * Aborts the VM if methods may not be invoked on this kind of value.
     */
    const VMDescriptor* find_dispatch_descriptor(const Value& value) const;

    /**
     * Find the descriptor of the value, for use in pattern matching.
     *
     * Returns null if the given value does not have a suitable descriptor (eg.
     * descriptors themselves don't have descriptors).
     */
    const VMDescriptor* find_match_descriptor(const Value& value) const;

    template<typename... Args>
    void trace(std::string_view fmt, Args&&... args) const
    {
      if (verbose_)
      {
        size_t indent = cfstack_.empty() ? 0 : cfstack_.size() - 1;
        fmt::print(std::cerr, "[{:4x}]: {:<{}}", start_ip_, "", indent);
        fmt::print(std::cerr, fmt, std::forward<Args>(args)...);
        fmt::print(std::cerr, "\n");
      }
    }

    template<typename... Args>
    [[noreturn]] void fatal(std::string_view fmt, Args&&... args) const
    {
      size_t indent = cfstack_.empty() ? 0 : cfstack_.size() - 1;
      fmt::print(std::cerr, "[{:4x}]: {:<{}}FATAL: ", start_ip_, "", indent);
      fmt::print(std::cerr, fmt, std::forward<Args>(args)...);
      fmt::print(std::cerr, "\n");
      abort();
    }

    void check_type(const Value& value, Value::Tag expected);
    void check_type(const Value& value, std::vector<Value::Tag> expected);

    const Code& code_;
    rt::Alloc* const alloc_;
    const bool verbose_;

    /**
     * Address of the currently executing instruction.
     *
     * After an instruction is parsed, frame().ip points to the start of the
     * next instruction. start_ip_ is used for tracing, and to resolve relative
     * offsets.
     *
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
     *
     * Because of finalisers, the VM needs to support re-entrant invocations,
     * which may cause the stack to grow at unexpected times. We use a deque
     * rather than a vector to make sure references don't get invalidated when
     * this happens.
     */
    std::deque<Value> stack_;

    struct Frame
    {
      /**
       * Bytecode offset at which instruction data is fetched.
       *
       * This value changes as operands get parsed from the bytecode. This means
       * during execution of an opcode, it actually points to the next
       * instruction. start_ip_ should be used to get the offset of the
       * currently executing instruction.
       */
      size_t ip;

      /**
       * Base offset into the value stack.
       *
       * All register accesses are relative to the current frame's base
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

      /**
       * Determine the behaviour of the RETURN opcode within this frame.
       * If Halt, it will pop the frame and halt execution. If Continue,
       * it will pop the frame and continue executing the previous one.
       */
      OnReturn on_return;
    };

    /**
     * Call stack.
     *
     * On CALL, a new frame is push on here, and on RETURN the top of
     * the stack is popped, restoring the previous state.
     */
    std::vector<Frame> cfstack_;

    Frame& frame()
    {
      assert(!cfstack_.empty());
      return cfstack_.back();
    }

    const Frame& frame() const
    {
      assert(!cfstack_.empty());
      return cfstack_.back();
    }

    template<typename T>
    friend struct convert_operand;
    template<typename T>
    friend struct execute_handler;

    template<bool IsConst>
    friend class BaseValueList;
  };

  /**
   * This represent the closure for all when clauses in the runtime
   */
  class ExecuteMessage : public rt::VBehaviour<ExecuteMessage>
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
