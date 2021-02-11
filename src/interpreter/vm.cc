// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "interpreter/vm.h"

#include "interpreter/convert.h"
#include "interpreter/format.h"
#include "interpreter/value_list.h"

#include <fmt/ranges.h>

namespace verona::interpreter
{
  void VM::run(std::vector<Value> args, size_t cown_count, size_t start)
  {
    assert(cfstack_.empty());

    halt_ = false;
    push_frame(start, 0, OnReturn::Halt);

    assert(static_cast<size_t>(frame().argc) == args.size());

    // First argument is the receiver, followed by cown_count cowns that are
    // being acquired, followed by captures. The cowns need to be transformed
    // so we actually pass their contents to the behaviour instead.
    for (size_t i = 0; i < args.size(); i++)
    {
      Register reg(truncate<uint8_t>(i));
      Value& value = args[i];
      if (i > 0 && i <= cown_count)
      {
        write(reg, value.cown_body());
        value.clear(alloc_);
      }
      else
      {
        write(reg, std::move(value));
      }
    }

    dispatch_loop();
  }

  void VM::push_frame(size_t ip, size_t base, OnReturn on_return)
  {
    FunctionHeader header = code_.function_header(ip);

    start_ip_ = ip;
    trace(
      "Calling function {}, base={:d}, argc={:d} retc={:d} locals={:d}",
      header.name,
      base,
      header.argc,
      header.retc,
      header.locals);

    Frame frame;
    frame.ip = ip;
    frame.argc = header.argc;
    frame.retc = header.retc;
    frame.locals = header.locals;
    frame.base = base;
    frame.on_return = on_return;

    grow_stack(frame.base + frame.locals);
    cfstack_.push_back(frame);
  }

  void VM::dispatch_loop()
  {
    while (!halt_)
    {
      start_ip_ = frame().ip;
      Opcode op = code_.load<Opcode>(frame().ip);
      dispatch_opcode(op);
    }
  }

  void VM::execute_finaliser(VMObject* object)
  {
    // This function gets called by the runtime to execute a finaliser.
    // We can't assume much about the VM's state when this is called, and we
    // must restore anything we might have tampered with.
    //
    // For example the finaliser could be called on a running VM as a
    // consequence of clearing a Register, or it could be called on a halted VM
    // as a consequence of the scheduler collection a cown.

    const VMDescriptor* descriptor = object->descriptor();
    assert(descriptor->finaliser_ip > 0);

    auto vm = VM::local_vm;

    // Save any VM state that isn't in stacks, and setup the VM into some
    // reasonable state.
    bool old_halt = std::exchange(vm->halt_, false);
    size_t old_start_ip =
      std::exchange(vm->start_ip_, descriptor->finaliser_ip);

    vm->trace("Running the finaliser for: {}", descriptor->name);

    // Set up a new frame for the finaliser. The frame starts past the current,
    // with no overlap, or at zero if there are no executing frames.
    //
    // The frame is marked as "halt on return" so we gain back control when the
    // finaliser is done.
    size_t base;
    if (vm->cfstack_.empty())
      base = 0;
    else
      base = vm->frame().base + vm->frame().locals;

    vm->push_frame(descriptor->finaliser_ip, base, OnReturn::Halt);

    if (vm->frame().argc != 1)
    {
      vm->fatal("Finaliser must have one argument, found {}", vm->frame().argc);
    }

    vm->write(Register(0), Value::mut(object));

    // Run finaliser to completion.
    vm->dispatch_loop();

    vm->halt_ = old_halt;
    vm->start_ip_ = old_start_ip;
  }

  void VM::grow_stack(size_t size)
  {
    size = snmalloc::bits::next_pow2(size);
    if (stack_.size() < size)
      stack_.resize(size);
  }

  Value& VM::read(Register reg)
  {
    if (reg.value >= frame().locals)
    {
      fatal("Out of bounds stack access (register {})", reg.value);
    }
    return stack_.at(frame().base + reg.value);
  }

  const Value& VM::read(Register reg) const
  {
    if (reg.value >= frame().locals)
    {
      fatal("Out of bounds stack access (register {})", reg.value);
    }
    return stack_.at(frame().base + reg.value);
  }

  void VM::write(Register reg, Value value)
  {
    if (reg.value >= frame().locals)
      fatal("Out of bounds stack access (register {})", reg.value);

    stack_.at(frame().base + reg.value).overwrite(alloc_, std::move(value));
  }

  const VMDescriptor* VM::find_dispatch_descriptor(const Value& value) const
  {
    switch (value.tag)
    {
      case Value::MUT:
      case Value::IMM:
      case Value::ISO:
        return value->object->descriptor();
      case Value::DESCRIPTOR:
        return value->descriptor;
      case Value::COWN:
      case Value::COWN_UNOWNED:
        return value->cown->descriptor;
      case Value::U64:
        return code_.special_descriptors().u64;
      case Value::STRING:
        return code_.special_descriptors().string;
      default:
        fatal("Cannot call method on {}", value);
    }
  }

  const VMDescriptor* VM::find_match_descriptor(const Value& value) const
  {
    switch (value.tag)
    {
      case Value::MUT:
      case Value::IMM:
      case Value::ISO:
        return value->object->descriptor();

      case Value::COWN:
        return value->cown->descriptor;

      case Value::U64:
        return code_.special_descriptors().u64;

      case Value::STRING:
        return code_.special_descriptors().string;

      case Value::DESCRIPTOR:
      case Value::UNINIT:
        return nullptr;

      default:
        fatal("Invalid match operand: {}", value);
    }
  }

  void VM::check_type(const Value& value, Value::Tag expected)
  {
    if (value.tag != expected)
      fatal(
        "Invalid tag {} for value {}, expected {}", value.tag, value, expected);
  }

  void VM::check_type(const Value& value, std::vector<Value::Tag> expected)
  {
    if (
      std::find(expected.begin(), expected.end(), value.tag) == expected.end())
    {
      fatal(
        "Invalid tag {} for value {}, expected one of {}",
        value.tag,
        value,
        expected);
    }
  }

  Value
  VM::opcode_binop(bytecode::BinaryOperator op, uint64_t left, uint64_t right)
  {
    switch (op)
    {
      case bytecode::BinaryOperator::Add:
        return Value::u64(left + right);
      case bytecode::BinaryOperator::Sub:
        return Value::u64(left - right);
      case bytecode::BinaryOperator::Mul:
        return Value::u64(left * right);
      case bytecode::BinaryOperator::Div:
        if (right == 0)
          fatal("Division by zero");
        return Value::u64(left / right);
      case bytecode::BinaryOperator::Mod:
        if (right == 0)
          fatal("Division by zero");
        return Value::u64(left % right);
      case bytecode::BinaryOperator::Shl:
        return Value::u64(left << right);
      case bytecode::BinaryOperator::Shr:
        return Value::u64(left >> right);
      case bytecode::BinaryOperator::Lt:
        return Value::u64(left < right);
      case bytecode::BinaryOperator::Gt:
        return Value::u64(left > right);
      case bytecode::BinaryOperator::Le:
        return Value::u64(left <= right);
      case bytecode::BinaryOperator::Ge:
        return Value::u64(left >= right);
      case bytecode::BinaryOperator::Eq:
        return Value::u64(left == right);
      case bytecode::BinaryOperator::Ne:
        return Value::u64(left != right);
      case bytecode::BinaryOperator::And:
        return Value::u64(left && right);
      case bytecode::BinaryOperator::Or:
        return Value::u64(left || right);

        EXHAUSTIVE_SWITCH;
    }
  }

  void VM::opcode_call(SelectorIdx selector, uint8_t callspace)
  {
    if (callspace == 0)
      fatal("Not enough call space to find a receiver");
    if (callspace > frame().locals)
      fatal("Call space does not fit in current frame");

    // Dispatch on the receiver, which is the first value in the callspace.
    const Value& receiver = read(Register(frame().locals - callspace));
    const VMDescriptor* descriptor = find_dispatch_descriptor(receiver);

    size_t addr = descriptor->methods[selector.value];
    size_t base = frame().base + frame().locals - callspace;

    push_frame(addr, base, OnReturn::Continue);

    if (callspace < frame().argc || callspace < frame().retc)
    {
      fatal(
        "Call space is too small: callspace={:d}, argc={:d}, retc={:d}",
        callspace,
        frame().argc,
        frame().retc);
    }
  }

  Value VM::opcode_clear()
  {
    return Value();
  }

  void VM::opcode_clear_list(ValueList values)
  {
    for (Value& value : values)
    {
      value.clear(alloc_);
    }
  }

  void VM::opcode_fulfill_sleeping_cown(const Value& cown, Value result)
  {
    check_type(cown, Value::COWN);

    cown->cown->contents = result.consume_iso();

    rt::Cown::acquire(cown->cown);
    cown->cown->schedule();
  }

  Value VM::opcode_freeze(Value src)
  {
    check_type(src, Value::ISO);

    VMObject* contents = src.consume_iso();
    rt::Freeze::apply(alloc_, contents);
    return Value::imm(contents);
  }

  Value VM::opcode_copy(Value src)
  {
    return std::move(src);
  }

  Value VM::opcode_int64(uint64_t imm)
  {
    return Value::u64(imm);
  }

  Value VM::opcode_string(std::string_view imm)
  {
    return Value::string(imm);
  }

  void VM::opcode_jump(RelativeOffset offset)
  {
    frame().ip = start_ip_ + offset.value;
  }

  void VM::opcode_jump_if(uint64_t condition, RelativeOffset offset)
  {
    if (condition > 0)
      frame().ip = start_ip_ + offset.value;
  }

  Value VM::opcode_load(const Value& base, SelectorIdx selector)
  {
    check_type(base, {Value::ISO, Value::MUT, Value::IMM});

    VMObject* object = base->object;
    const VMDescriptor* descriptor = object->descriptor();
    size_t index = descriptor->fields[selector.value];

    Value value = object->fields[index].read(base.tag);
    return std::move(value);
  }

  Value VM::opcode_load_descriptor(DescriptorIdx desc_idx)
  {
    const VMDescriptor* descriptor = code_.get_descriptor(desc_idx);
    return Value::descriptor(descriptor);
  }

  Value VM::opcode_match_descriptor(const Value& src, const VMDescriptor* desc)
  {
    const VMDescriptor* src_descriptor = find_match_descriptor(src);

    uint64_t result;

    // Some values are unmatchable, in which case their descriptor is null.
    if (src_descriptor == nullptr)
      result = 0;
    else
      result = desc->subtypes.count(src_descriptor->index) > 0;

    trace(" Matching {} against {} = {}", src, desc->name, result);
    return Value::u64(result);
  }

  Value VM::opcode_match_capability(const Value& src, bytecode::Capability cap)
  {
    uint64_t result;
    switch (src.tag)
    {
      case Value::ISO:
        result = (cap == bytecode::Capability::Iso);
        break;

      case Value::MUT:
        result = (cap == bytecode::Capability::Mut);
        break;

      // These are all represented as immutables in the source language, even if
      // we don't actually implement them as immutable objects in the VM.
      case Value::IMM:
      case Value::COWN:
      case Value::U64:
      case Value::STRING:
        result = (cap == bytecode::Capability::Imm);
        break;

      default:
        result = 0;
        break;
    }

    trace(" Matching {} against {} = {}", src, cap, result);
    return Value::u64(result);
  }

  Value VM::opcode_move(Register src)
  {
    return std::move(read(src));
  }

  Value VM::opcode_mut_view(const Value& src)
  {
    check_type(src, {Value::ISO, Value::MUT});

    return Value::mut(src->object);
  }

  Value
  VM::opcode_new_object(const Value& parent, const VMDescriptor* descriptor)
  {
    check_type(parent, {Value::ISO, Value::MUT});

    VMObject* region = parent->object->region();
    rt::Object* object = rt::Region::alloc(alloc_, region, descriptor);
    return Value::mut(new (object) VMObject(region, descriptor));
  }

  Value VM::opcode_new_region(const VMDescriptor* descriptor)
  {
    // TODO(region): For now, the only kind of region we can create is a trace
    // region. Later, we might need a new bytecode?
    rt::Object* object = rt::RegionTrace::create(alloc_, descriptor);
    return Value::iso(new (object) VMObject(nullptr, descriptor));
  }

  Value VM::opcode_new_cown(const VMDescriptor* descriptor, Value src)
  {
    check_type(src, Value::ISO);
    VMObject* contents = src.consume_iso();
    return Value::cown(new VMCown(descriptor, contents));
  }

  Value VM::opcode_new_sleeping_cown(const VMDescriptor* descriptor)
  {
    auto a = Value::cown(new VMCown(descriptor));
    trace(" New sleeping cown {}", a);

    return std::move(a);
  }

  void VM::opcode_trace_region(const Value& object)
  {
    check_type(object, {Value::ISO, Value::MUT});

    VMObject* region = object->object->region();
    rt::RegionTrace::gc(alloc_, region);
  }

  void VM::opcode_print(std::string_view fmt, ConstValueList values)
  {
    fmt::dynamic_format_arg_store<fmt::format_context> store;
    for (const Value& value : values)
    {
      store.push_back(std::cref(value));
    }
    fmt::vprint(fmt, store);
  }

  void VM::opcode_return()
  {
    // Ensure that all registers (except the return values) have been cleared
    // already.
    for (int i = frame().retc; i < frame().locals; i++)
    {
      Value& value = read(Register(i));
      switch (value.tag)
      {
        case Value::UNINIT:
          break;

        case Value::DESCRIPTOR:
        case Value::U64:
          // Codegen creates some DESCRIPTOR and U64 values which don't exist in
          // the IR and thus have no corresponding end-scope statement. Ideally
          // codegen should be cleaning this up, but not yet.
          value.clear(alloc_);
          break;

        default:
          fatal("Register {} was not cleared: {}", i, value);
          break;
      }
    }

    if (frame().on_return == OnReturn::Halt)
    {
      // We currently never use the return value of the top function, so just
      // clear the return registers.
      for (int i = frame().retc; i < frame().locals; i++)
      {
        read(Register(i)).clear(alloc_);
      }

      halt_ = true;
    }
    else if (cfstack_.size() < 2)
    {
      fatal("Cannot return of top-most frame");
    }
    cfstack_.pop_back();
  }

  Value VM::opcode_store(const Value& base, SelectorIdx selector, Value src)
  {
    check_type(base, {Value::ISO, Value::MUT});

    VMObject* object = base->object;
    const VMDescriptor* desc = object->descriptor();
    size_t index = desc->fields[selector.value];

    if (src.tag == Value::Tag::MUT && object->region() != src->object->region())
    {
      fatal("Writing reference to incorrect region");
    }

    Value old_value =
      object->fields[index].exchange(alloc_, object->region(), std::move(src));
    return std::move(old_value);
  }

  void VM::opcode_when(
    AbsoluteOffset offset, uint8_t cown_count, uint8_t capture_count)
  {
    // One added for unused receiver
    // TODO-Better-Static-codegen
    auto callspace = cown_count + capture_count + 1;
    if (callspace > frame().locals)
      fatal("Call space does not fit in current frame");

    size_t addr = offset.value;
    FunctionHeader header = code_.function_header(addr);

    if (callspace > header.argc)
    {
      fatal(
        "Incorrect ABI for `when`: callspace={:d}, argc={:d}",
        callspace,
        header.argc);
    }

    // Compute the base of the new "frame", relative to the current frame.
    // We use this to copy these values into the message
    size_t base = frame().locals - callspace;

    // Prepare the cowns and the arguments for the method invocation.
    std::vector<Value> args;
    std::vector<rt::Cown*> cowns;
    args.reserve(header.argc);
    cowns.reserve(cown_count);

    // First argument is a placeholder for the receiver.
    args.push_back(Value());

    // The rest are the cowns
    for (size_t i = 0; i < cown_count; i++)
    {
      Value& v = read(Register(truncate<uint8_t>(base + 1 + i)));
      trace("Capturing cown {:d}: {}", i, v);
      check_type(v, Value::COWN);

      // Push the body of the cown into the message, as an unowned cown. The
      // runtime will be holding a reference to the cown for us, so no need to
      // have our own.
      //
      // We can't look up the pointer to the cown's contents, since for promise
      // cowns it is not set until the promise is fulfilled.
      args.push_back(v.as_unowned_cown());

      // Transfer ownership of the cown from `v` into the `cowns` vector. The
      // runtime will hold on to the references until after the message is
      // executed.
      cowns.push_back(v.consume_cown());
    }

    // The rest are the captured values
    for (size_t i = 0; i < capture_count; i++)
    {
      Value& v = read(Register(truncate<uint8_t>(base + 1 + cown_count + i)));
      trace("Capturing variable {:d}: {}", i + cown_count, v);
      args.push_back(std::move(v));
    }

    trace(
      "Dispatching when to function {}, argc={:d}", header.name, header.argc);

    // If no cowns create a fake one to run the code on.
    if (cowns.size() == 0)
    {
      cowns.push_back(new VMCown(nullptr, nullptr));
    }

    rt::Cown::schedule<ExecuteMessage, rt::YesTransfer>(
      cowns.size(), cowns.data(), offset.value, std::move(args), cown_count);
  }

  void VM::opcode_protect(ConstValueList values)
  {
    for (const Value& value : values)
    {
      // Only MUTs need to be protected against GC. ISOs are the entrypoint to
      // the region, hence are always traced. IMM and COWNs hold reference
      // counts to their object. The rest aren't managed by the runtime.
      if (value.tag == Value::MUT)
      {
        VMObject* object = value->object;
        VMObject* region = object->region();
        rt::RegionTrace::push_additional_root(region, object, alloc_);
      }
    }
  }

  void VM::opcode_unprotect(ConstValueList values)
  {
    // We iterate over the values in reverse order, in accordance with the stack
    // API exposed by the runtime.
    for (auto it = values.rbegin(); it != values.rend(); ++it)
    {
      const Value& value = *it;
      if (value.tag == Value::MUT)
      {
        VMObject* object = value->object;
        VMObject* region = object->region();
        rt::RegionTrace::pop_additional_root(region, object, alloc_);
      }
    }
  }

  void VM::opcode_unreachable()
  {
    fatal("Reached unreachable opcode");
  }

  void VM::dispatch_opcode(Opcode op)
  {
    switch (op)
    {
#define OP(NAME, FN) \
  case Opcode::NAME: \
    execute_opcode<Opcode::NAME, &VM::FN>(frame().ip); \
    break;

      OP(BinOp, opcode_binop);
      OP(Call, opcode_call);
      OP(Clear, opcode_clear);
      OP(ClearList, opcode_clear_list);
      OP(Copy, opcode_copy);
      OP(FulfillSleepingCown, opcode_fulfill_sleeping_cown);
      OP(Freeze, opcode_freeze);
      OP(Int64, opcode_int64);
      OP(Jump, opcode_jump);
      OP(JumpIf, opcode_jump_if);
      OP(Load, opcode_load);
      OP(LoadDescriptor, opcode_load_descriptor);
      OP(MatchCapability, opcode_match_capability);
      OP(MatchDescriptor, opcode_match_descriptor);
      OP(Move, opcode_move);
      OP(MutView, opcode_mut_view);
      OP(NewObject, opcode_new_object);
      OP(NewRegion, opcode_new_region);
      OP(NewSleepingCown, opcode_new_sleeping_cown);
      OP(NewCown, opcode_new_cown);
      OP(Print, opcode_print);
      OP(Protect, opcode_protect);
      OP(Return, opcode_return);
      OP(Store, opcode_store);
      OP(String, opcode_string);
      OP(TraceRegion, opcode_trace_region);
      OP(When, opcode_when);
      OP(Unprotect, opcode_unprotect);
      OP(Unreachable, opcode_unreachable);

#undef OP

      default:
        fatal("Invalid opcode {:#x}", static_cast<int>(op));
    }
  }

  template<Opcode opcode, auto Fn>
  void VM::execute_opcode(size_t& ip)
  {
    static_assert(std::is_member_function_pointer_v<decltype(Fn)>);

    auto operands = code_.load_operands<opcode>(ip);

    // The std::apply with a lambda trick turns the operands tuple into a
    // parameter pack, so it can more easily be used.
    std::apply(
      [&](const auto&... args) {
        this->trace(bytecode::OpcodeSpec<opcode>::format, args...);
        execute_handler<decltype(Fn)>::template execute<Fn>(this, args...);
      },
      operands);
  }
}
