// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "interpreter/vm.h"

#include "interpreter/convert.h"
#include "interpreter/format.h"

#include <fmt/ranges.h>

namespace verona::interpreter
{
  void VM::run(std::vector<Value> args, size_t cown_count, size_t start)
  {
    halt_ = false;
    start_ip_ = ip_ = start;

    std::string_view name = code_.str(ip_);

    frame_ = Frame::initial();
    frame_.argc = code_.u8(ip_);
    frame_.retc = code_.u8(ip_);
    frame_.locals = code_.u8(ip_);
    code_.u32(ip_); // size

    assert(static_cast<size_t>(frame_.argc) == args.size());

    trace(
      "Entering function {}, argc={:d} retc={:d} locals={:d}",
      name,
      frame_.argc,
      frame_.retc,
      frame_.locals);

    // Load registers with cowns values.
    assert(frame_.base == 0);

    // Ensure the stack is large enough.
    grow_stack(frame_.base + frame_.argc + frame_.locals);

    size_t index = 0;

    // First argument is the receiver, followed by cown_count cowns that are
    // being acquired, followed by captures.
    for (auto& a : args)
    {
      if (index > 0 && index <= cown_count)
      {
        a.switch_to_cown_body();
      }
      stack_.at(index).overwrite(alloc_, std::move(a));

      index++;
    }

    dispatch_loop();
  }

  void VM::dispatch_loop()
  {
    while (!halt_)
    {
      start_ip_ = ip_;
      Opcode op = code_.opcode(ip_);
      dispatch_opcode(op);
    }
  }

  void VM::execute_finaliser(VMObject* object)
  {
    const VMDescriptor* descriptor = object->descriptor();
    assert(descriptor->finaliser_ip > 0);

    auto vm = VM::local_vm;
    vm->trace("Finaliser for: {}", descriptor->name);

    auto old_halt = vm->halt_;
    vm->halt_ = false;

    // Set up a new frame for finaliser
    vm->write(Register(vm->frame_.locals - 1), Value::mut(object));
    vm->call(descriptor->finaliser_ip, (uint8_t)1);

    // Save call stack, so we can jump back into normal execution.
    auto backup = std::move(vm->cfstack_);

    // Run finaliser to completion
    vm->dispatch_loop();

    // Put back normal execution
    vm->cfstack_ = std::move(backup);
    vm->opcode_return();
    vm->halt_ = old_halt;
  }

  void VM::grow_stack(size_t size)
  {
    size = snmalloc::bits::next_pow2(size);
    if (stack_.size() < size)
      stack_.resize(size);
  }

  Value& VM::read(Register reg)
  {
    if (reg.index >= frame_.locals)
      fatal("Out of bounds stack access (register {})", reg.index);
    return stack_.at(frame_.base + reg.index);
  }

  const Value& VM::read(Register reg) const
  {
    if (reg.index >= frame_.locals)
      fatal("Out of bounds stack access (register {})", reg.index);
    return stack_.at(frame_.base + reg.index);
  }

  void VM::write(Register reg, Value value)
  {
    if (reg.index >= frame_.locals)
      fatal("Out of bounds stack access (register {})", reg.index);

    stack_.at(frame_.base + reg.index).overwrite(alloc_, std::move(value));
  }

  const VMDescriptor* VM::find_dispatch_descriptor(Register receiver) const
  {
    const Value& value = read(receiver);
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
      default:
        fatal("Cannot call method on {}={}", receiver, value);
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
    if (callspace > frame_.locals)
      fatal("Call space does not fit in current frame");

    // Dispatch on the receiver, which is the first value in the callspace.
    const VMDescriptor* descriptor =
      find_dispatch_descriptor(Register(frame_.locals - callspace));

    size_t addr = descriptor->methods[selector];
    VM::call(addr, callspace);
  }

  void VM::call(size_t addr, uint8_t callspace)
  {
    FunctionHeader header = code_.function_header(addr);

    if (callspace < header.argc || callspace < header.retc)
    {
      fatal(
        "Call space is too small: callspace={:d}, argc={:d}, retc={:d}",
        callspace,
        header.argc,
        header.retc);
    }

    // End of current frame
    size_t top = frame_.base + frame_.locals;

    cfstack_.push_back(frame_);
    indent_++;

    frame_.argc = header.argc;
    frame_.retc = header.retc;
    frame_.locals = header.locals;
    frame_.base = top - callspace;
    frame_.return_address = ip_;

    // Ensure the stack is large enough.
    grow_stack(frame_.base + frame_.locals);

    ip_ = addr;

    trace(
      "Calling function {}, base=r{:d} argc={:d} retc={:d} locals={:d}",
      header.name,
      frame_.base,
      frame_.argc,
      frame_.retc,
      frame_.locals);
  }

  Value VM::opcode_clear()
  {
    return Value();
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

  void VM::opcode_error(std::string_view reason)
  {
    fatal("{}", reason);
  }

  Value VM::opcode_int64(uint64_t imm)
  {
    return Value::u64(imm);
  }

  Value VM::opcode_string(std::string_view imm)
  {
    return Value::string(imm);
  }

  void VM::opcode_jump(int16_t offset)
  {
    ip_ = start_ip_ + offset;
  }

  void VM::opcode_jump_if(uint64_t condition, int16_t offset)
  {
    if (condition > 0)
      ip_ = start_ip_ + offset;
  }

  Value VM::opcode_load(const Value& base, SelectorIdx selector)
  {
    check_type(base, {Value::ISO, Value::MUT, Value::IMM});

    VMObject* object = base->object;
    const VMDescriptor* descriptor = object->descriptor();
    size_t index = descriptor->fields[selector];

    Value value = object->fields[index].read(base.tag);
    return std::move(value);
  }

  Value VM::opcode_load_descriptor(DescriptorIdx desc_idx)
  {
    const VMDescriptor* descriptor = code_.get_descriptor(desc_idx);
    return Value::descriptor(descriptor);
  }

  Value VM::opcode_match(const Value& src, const VMDescriptor* descriptor)
  {
    uint64_t result;
    switch (src.tag)
    {
      case Value::UNINIT:
      case Value::U64:
      case Value::STRING:
      case Value::DESCRIPTOR:
      case Value::COWN:
        result = false;
        break;

      case Value::ISO:
      case Value::IMM:
      case Value::MUT:
        result = (src->object->descriptor() == descriptor);
        break;
      case Value::COWN_UNOWNED:
        // This type should only appear in message.
        abort();
    }

    trace(" Matching {} against {} = {}", src, descriptor->name, result);

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
    return Value::mut(new (object) VMObject(region));
  }

  Value VM::opcode_new_region(const VMDescriptor* descriptor)
  {
    // TODO(region): For now, the only kind of region we can create is a trace
    // region. Later, we might need a new bytecode?
    rt::Object* object = rt::RegionTrace::create(alloc_, descriptor);
    return Value::iso(new (object) VMObject(nullptr));
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

  void VM::opcode_print(std::string_view fmt, uint8_t argc)
  {
    fmt::dynamic_format_arg_store<fmt::format_context> store;
    for (uint8_t i = 0; i < argc; i++)
    {
      Register reg = code_.load<Register>(ip_);
      store.push_back(std::cref(read(reg)));
    }
    fmt::vprint(fmt, store);
  }

  void VM::opcode_return()
  {
    // Ensure that all registers (except the return values) have been cleared
    // already.
    for (int i = frame_.retc; i < frame_.locals; i++)
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

    if (cfstack_.empty())
    {
      // We currently never use the return value of the top function, so just
      // clear the return registers.
      for (int i = frame_.retc; i < frame_.locals; i++)
      {
        read(Register(i)).clear(alloc_);
      }

      halt_ = true;
      return;
    }

    ip_ = frame_.return_address;
    frame_ = cfstack_.back();
    cfstack_.pop_back();
    indent_--;
  }

  Value VM::opcode_store(const Value& base, SelectorIdx selector, Value src)
  {
    check_type(base, {Value::ISO, Value::MUT});

    VMObject* object = base->object;
    const VMDescriptor* desc = object->descriptor();
    size_t index = desc->fields[selector];

    if (src.tag == Value::Tag::MUT && object->region() != src->object->region())
    {
      fatal("Writing reference to incorrect region");
    }

    Value old_value =
      object->fields[index].exchange(alloc_, object->region(), std::move(src));
    return std::move(old_value);
  }

  void VM::opcode_when(
    CodePtr closure_body, uint8_t cown_count, uint8_t capture_count)
  {
    // One added for unused receiver
    // TODO-Better-Static-codegen
    auto callspace = cown_count + capture_count + 1;
    if (callspace > frame_.locals)
      fatal("Call space does not fit in current frame");

    size_t entry_addr = closure_body;
    size_t addr = entry_addr;
    FunctionHeader header = code_.function_header(addr);

    if (callspace > header.argc)
    {
      fatal(
        "Incorrect ABI for `when`: callspace={:d}, argc={:d}",
        callspace,
        header.argc);
    }

    size_t top = frame_.base + frame_.locals;
    Value* values = &stack_[top - callspace + 1];

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
      Value& v = values[i];
      trace("Capturing cown {:d}: {}", i, v);
      check_type(v, Value::COWN);

      // Multimessage will take increfs on all the cowns, so don't need to
      // protect them here.
      cowns.push_back(v->cown);
      // Releases reference count to caller, so we can use it inside
      // multimessage.
      v.consume_cown();
      args.push_back(std::move(v));
    }

    // The rest are the captured values
    for (size_t i = 0; i < capture_count; i++)
    {
      Value& v = values[i + cown_count];
      trace("Capturing variable {:d}: {}", i + cown_count, v);
      args.push_back(std::move(v));
    }

    trace(
      "Dispatching when to function {}, argc={:d}", header.name, frame_.argc);

    // If no cowns create a fake one to run the code on.
    if (cowns.size() == 0)
    {
      cowns.push_back(new VMCown(nullptr, nullptr));
    }

    rt::Cown::schedule<ExecuteMessage, rt::YesTransfer>(
      cowns.size(), cowns.data(), entry_addr, std::move(args), cown_count);
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
    execute_opcode<Opcode::NAME, &VM::FN>(ip_); \
    break;

      OP(BinOp, opcode_binop);
      OP(Call, opcode_call);
      OP(Clear, opcode_clear);
      OP(Copy, opcode_copy);
      OP(FulfillSleepingCown, opcode_fulfill_sleeping_cown);
      OP(Error, opcode_error);
      OP(Freeze, opcode_freeze);
      OP(Int64, opcode_int64);
      OP(Jump, opcode_jump);
      OP(JumpIf, opcode_jump_if);
      OP(Load, opcode_load);
      OP(LoadDescriptor, opcode_load_descriptor);
      OP(Match, opcode_match);
      OP(Move, opcode_move);
      OP(MutView, opcode_mut_view);
      OP(NewObject, opcode_new_object);
      OP(NewRegion, opcode_new_region);
      OP(NewSleepingCown, opcode_new_sleeping_cown);
      OP(NewCown, opcode_new_cown);
      OP(Print, opcode_print);
      OP(Return, opcode_return);
      OP(Store, opcode_store);
      OP(String, opcode_string);
      OP(TraceRegion, opcode_trace_region);
      OP(When, opcode_when);
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
