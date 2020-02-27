// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "interpreter/vm.h"

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

  void VM::opcode_binop(
    Register dst,
    bytecode::BinaryOperator op,
    const Value& left,
    const Value& right)
  {
    check_type(left, Value::Tag::U64);
    check_type(right, Value::Tag::U64);

    uint64_t result;

    switch (op)
    {
      case bytecode::BinaryOperator::Add:
        result = left->u64 + right->u64;
        break;
      case bytecode::BinaryOperator::Sub:
        result = left->u64 - right->u64;
        break;
      case bytecode::BinaryOperator::Lt:
        result = left->u64 < right->u64;
        break;
      case bytecode::BinaryOperator::Gt:
        result = left->u64 > right->u64;
        break;
      case bytecode::BinaryOperator::Le:
        result = left->u64 <= right->u64;
        break;
      case bytecode::BinaryOperator::Ge:
        result = left->u64 >= right->u64;
        break;
      case bytecode::BinaryOperator::Eq:
        result = left->u64 == right->u64;
        break;
      case bytecode::BinaryOperator::Ne:
        result = left->u64 != right->u64;
        break;
      case bytecode::BinaryOperator::And:
        result = left->u64 && right->u64;
        break;
      case bytecode::BinaryOperator::Or:
        result = left->u64 || right->u64;
        break;

        EXHAUSTIVE_SWITCH;
    }

    write(dst, Value::u64(result));
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

  void VM::opcode_clear(Register dst)
  {
    write(dst, Value());
  }

  void VM::opcode_fulfill_sleeping_cown(const Value& cown, Value result)
  {
    check_type(cown, Value::Tag::COWN);

    cown->cown->contents = result.consume_iso();

    rt::Cown::acquire(cown->cown);
    cown->cown->schedule();
  }

  void VM::opcode_freeze(Register dst, Value src)
  {
    check_type(src, Value::Tag::ISO);

    VMObject* contents = src.consume_iso();
    rt::Freeze::apply(alloc_, contents);
    write(dst, Value::imm(contents));
  }

  void VM::opcode_copy(Register dst, Value src)
  {
    write(dst, std::move(src));
  }

  void VM::opcode_int64(Register dst, uint64_t imm)
  {
    write(dst, Value::u64(imm));
  }

  void VM::opcode_string(Register dst, std::string_view imm)
  {
    write(dst, Value::string(imm));
  }

  void VM::opcode_jump(int16_t offset)
  {
    ip_ = start_ip_ + offset;
  }

  void VM::opcode_jump_if(const Value& src, int16_t offset)
  {
    check_type(src, Value::Tag::U64);

    if (src->u64 > 0)
      ip_ = start_ip_ + offset;
  }

  void VM::opcode_load(Register dst, const Value& base, SelectorIdx selector)
  {
    check_type(base, {Value::Tag::ISO, Value::Tag::MUT, Value::Tag::IMM});

    VMObject* object = base->object;
    const VMDescriptor* descriptor = object->descriptor();
    size_t index = descriptor->fields[selector];

    Value value = object->fields[index].read(base.tag);
    write(dst, std::move(value));
  }

  void VM::opcode_load_descriptor(Register dst, DescriptorIdx desc_idx)
  {
    const VMDescriptor* descriptor = code_.get_descriptor(desc_idx);
    write(dst, Value::descriptor(descriptor));
  }

  void VM::opcode_match(
    Register dst, const Value& src, const VMDescriptor* descriptor)
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

    write(dst, Value::u64(result));
  }

  void VM::opcode_move(Register dst, Register src)
  {
    write(dst, std::move(read(src)));
  }

  void VM::opcode_mut_view(Register dst, const Value& src)
  {
    check_type(src, {Value::Tag::ISO, Value::Tag::MUT});

    write(dst, Value::mut(src->object));
  }

  void VM::opcode_new(
    Register dst, const Value& parent, const VMDescriptor* descriptor)
  {
    check_type(parent, {Value::Tag::ISO, Value::Tag::MUT});

    VMObject* region = parent->object->region();
    rt::Object* object = rt::Region::alloc(alloc_, region, descriptor);
    write(dst, Value::mut(new (object) VMObject(region)));
  }

  void VM::opcode_new_region(Register dst, const VMDescriptor* descriptor)
  {
    // TODO(region): For now, the only kind of region we can create is a trace
    // region. Later, we might need a new bytecode?
    rt::Object* object = rt::RegionTrace::create(alloc_, descriptor);
    write(dst, Value::iso(new (object) VMObject(nullptr)));
  }

  void
  VM::opcode_new_cown(Register dst, const VMDescriptor* descriptor, Value src)
  {
    check_type(src, Value::Tag::ISO);
    VMObject* contents = src.consume_iso();
    write(dst, Value::cown(new VMCown(descriptor, contents)));
  }

  void
  VM::opcode_new_sleeping_cown(Register dst, const VMDescriptor* descriptor)
  {
    auto a = Value::cown(new VMCown(descriptor));
    trace(" New sleeping cown {}", a);

    write(dst, std::move(a));
  }

  void VM::opcode_trace_region(const Value& object)
  {
    check_type(object, {Value::Tag::ISO, Value::Tag::MUT});

    VMObject* region = object->object->region();

    rt::RegionTrace::gc(alloc_, region);
  }

  void VM::opcode_print(const Value& src, uint8_t argc)
  {
    check_type(src, Value::Tag::STRING);
    std::string_view format = src->string();

    std::vector<const Value*> values;
    for (uint8_t i = 0; i < argc; i++)
    {
      values.push_back(&read(code_.load<Register>(ip_)));
    }

    // Sadly fmt doesn't have any public API for dynamic sized lists of
    // arguments.
    switch (argc)
    {
      case 0:
        fmt::print(format);
        break;
      case 1:
        fmt::print(format, *values[0]);
        break;
      case 2:
        fmt::print(format, *values[0], *values[1]);
        break;
      case 3:
        fmt::print(format, *values[0], *values[1], *values[2]);
        break;
      case 4:
        fmt::print(format, *values[0], *values[1], *values[2], *values[3]);
        break;
      case 5:
        fmt::print(
          format, *values[0], *values[1], *values[2], *values[3], *values[4]);
        break;
      default:
        fatal("{} is more arguments than opcode_print can handle", argc);
    };
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

  void VM::opcode_store(
    Register dst, const Value& base, SelectorIdx selector, Value src)
  {
    check_type(base, {Value::Tag::ISO, Value::Tag::MUT});

    VMObject* object = base->object;
    const VMDescriptor* desc = object->descriptor();
    size_t index = desc->fields[selector];

    if (src.tag == Value::Tag::MUT && object->region() != src->object->region())
    {
      fatal("Writing reference to incorrect region");
    }

    Value old_value =
      object->fields[index].exchange(alloc_, object->region(), std::move(src));
    write(dst, std::move(old_value));
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
      check_type(v, Value::Tag::COWN);

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
      OP(Freeze, opcode_freeze);
      OP(Int64, opcode_int64);
      OP(Jump, opcode_jump);
      OP(JumpIf, opcode_jump_if);
      OP(Load, opcode_load);
      OP(LoadDescriptor, opcode_load_descriptor);
      OP(Match, opcode_match);
      OP(Move, opcode_move);
      OP(MutView, opcode_mut_view);
      OP(New, opcode_new);
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
    std::apply(
      [&](const auto&... args) {
        this->trace(bytecode::OpcodeSpec<opcode>::format, args...);
      },
      operands);

    auto arguments =
      convert_operand_list<decltype(Fn)>::convert(this, operands);
    std::apply(
      [&](auto&&... args) {
        (this->*Fn)(std::forward<decltype(args)>(args)...);
      },
      std::move(arguments));
  }
}
