#include "thread.h"

#include "array.h"
#include "callback.h"
#include "cown.h"
#include "drag.h"
#include "freeze.h"
#include "function_signature.h"
#include "merge.h"
#include "object.h"
#include "program.h"
#include "region_ext.h"

#include <cstdlib>
#include <iostream>
#include <sched/schedulerthread.h>
#include <source_location>

#ifndef NDEBUG
#  include <algorithm>
#  include <unordered_map>
#endif

#define INLINE SNMALLOC_FAST_PATH_LAMBDA

namespace vbci
{
  template<typename>
  struct always_false : std::false_type
  {};

  // Forward declarations
  template<typename F, typename... Args>
  static void process(Thread& self, F fun, Args... args);

  struct ArgReg
  {
    Register& reg;

    void operator=(Register&& v)
    {
      reg = std::move(v);
    }

    void operator=(const Register& v)
    {
      reg = v;
    }
  };

  static bool ffi_ptr_compatible(const Register& arg)
  {
    switch (arg->type())
    {
      case ValueType::None:
      case ValueType::Ptr:
      case ValueType::Object:
      case ValueType::Array:
        return true;

      default:
        return false;
    }
  }

  static void marshal_ffi_ptr_arg(
    const Register& arg, const void*& ffi_arg_addr, const void*& ffi_arg_val)
  {
    switch (arg->type())
    {
      case ValueType::None:
        ffi_arg_val = nullptr;
        ffi_arg_addr = &ffi_arg_val;
        return;

      case ValueType::Object:
        ffi_arg_val = arg->get_object()->get_pointer();
        ffi_arg_addr = &ffi_arg_val;
        return;

      case ValueType::Array:
        ffi_arg_val = arg->get_array()->get_pointer();
        ffi_arg_addr = &ffi_arg_val;
        return;

      default:
        ffi_arg_addr = arg->to_ffi();
        return;
    }
  }

  static bool ffi_struct_leaf_type(
    Program& program, uint32_t type_id, ValueType& kind, ffi_type*& rep)
  {
    if (
      (type_id == DynamicTypeId) || program.is_tuple(type_id) ||
      program.is_ref(type_id) || program.is_union(type_id))
      return false;

    auto layout = program.layout_type_id(type_id);
    kind = layout.first;
    rep = layout.second;

    switch (kind)
    {
      case ValueType::Bool:
      case ValueType::I8:
      case ValueType::I16:
      case ValueType::I32:
      case ValueType::I64:
      case ValueType::U8:
      case ValueType::U16:
      case ValueType::U32:
      case ValueType::U64:
      case ValueType::ILong:
      case ValueType::ULong:
      case ValueType::ISize:
      case ValueType::USize:
      case ValueType::F32:
      case ValueType::F64:
      case ValueType::Ptr:
      case ValueType::Object:
      case ValueType::Array:
      case ValueType::Cown:
        return true;

      default:
        return false;
    }
  }

  static bool ffi_struct_members(
    Program& program,
    uint32_t layout_type_id,
    std::vector<ValueType>& kinds,
    std::vector<ffi_type*>& reps)
  {
    auto add_leaf = [&](uint32_t field_type_id) {
      ValueType kind;
      ffi_type* rep;
      if (!ffi_struct_leaf_type(program, field_type_id, kind, rep))
        return false;
      kinds.push_back(kind);
      reps.push_back(rep);
      return true;
    };

    if (program.is_tuple(layout_type_id))
    {
      auto& tuple = program.complex_type(layout_type_id);

      if (tuple.children.empty())
        return false;

      for (auto field_type_id : tuple.children)
      {
        if (program.is_tuple(field_type_id) || !add_leaf(field_type_id))
          return false;
      }

      return true;
    }

    return add_leaf(layout_type_id);
  }

  static ValueType
  ffi_expected_kind(Program& program, uint32_t expected_type_id)
  {
    ValueType kind;
    ffi_type* rep;

    if (!ffi_struct_leaf_type(program, expected_type_id, kind, rep))
      Value::error(Error::BadType);

    return kind;
  }

  static void* ffi_field_addr(const Register& base, const Register& offset)
  {
    auto* ptr = static_cast<uint8_t*>(base->get_ptr());

    if (ptr == nullptr)
      Value::error(Error::BadOperand);

    return ptr + offset->get_size();
  }

  struct ConstantBase
  {};
  template<typename T_>
  struct Constant : ConstantBase
  {
    using T = T_;
    T value;

    Constant(T v) : value(v) {}

    // Implicit conversion to T
    operator T() const
    {
      return value;
    }
  };

  struct Operands
  {
    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    call_function_and_continue(F fun, Args... args)
    {
      fun(std::forward<Args>(args)...);
    }

    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    register_ref(Thread& self, F fun, Args... args)
    {
      size_t reg_index = self.leb();
      Register& reg = self.frame->local(reg_index);
      self.trace_instruction(" Local=", reg_index, " (", reg, ")");
      process<F, Args..., Register&>(
        self, fun, std::forward<Args>(args)..., reg);
      LOG(Trace) << "Updated " << reg_index << " to " << reg;
    }

    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    register_borrow(Thread& self, F fun, Args... args)
    {
      size_t reg_index = self.leb();
      const Register& reg = self.frame->local(reg_index);
      self.trace_instruction(" Local=", reg_index, " (", reg, ")");
      process<F, Args..., const Register&>(
        self, fun, std::forward<Args>(args)..., reg);
    }

    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    register_move(Thread& self, F fun, Args... args)
    {
      size_t reg_index = self.leb();
      Register& reg = self.frame->local(reg_index);
      self.trace_instruction(" Local=", reg_index, " (", reg, ")");
      process<F, Args..., Register>(
        self, fun, std::forward<Args>(args)..., std::move(reg));
    }

    template<typename T, typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    load_constant_param(Thread& self, F fun, Args... args)
    {
      size_t constant_value = self.leb();
      self.trace_instruction(" Constant=", constant_value);
      Constant<T> c{static_cast<T>(constant_value)};
      process<F, Args..., Constant<T>>(
        self, fun, std::forward<Args>(args)..., c);
    }

    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    load_class_param(Thread& self, F fun, Args... args)
    {
      size_t class_id = self.leb();
      auto& cls = self.program->cls(class_id);
      self.trace_instruction(" Class=", class_id);
      process<F, Args..., Class&>(self, fun, std::forward<Args>(args)..., cls);
    }

    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void
    load_function_ptr_param(Thread& self, F fun, Args... args)
    {
      size_t func_id = self.leb();
      Function* func = self.program->function(func_id);
      self.trace_instruction(
        " FunctionPtr=", func_id, "(", self.program->di_function(func), ")");
      process<F, Args..., Function*>(
        self, fun, std::forward<Args>(args)..., func);
    }

    // Tail-recursive process passing the current parameter index
    template<typename F, typename... Args>
    SNMALLOC_FAST_PATH static void process(Thread& self, F fun, Args... args)
    {
      constexpr size_t current_param = sizeof...(Args);

      if constexpr (param_count_v<F> == current_param)
      {
#ifndef NDEBUG
        self.instruction_log.dump_and_reset(logging::detail::LogLevel::Trace);
#endif
        call_function_and_continue<F, Args...>(
          fun, std::forward<Args>(args)...);
#ifndef NDEBUG
        logging::Log log(logging::detail::LogLevel::Trace);
        self.print_stack(log, true);
#endif
      }
      else
      {
        using T = param_type_t<F, current_param>;
        if constexpr (std::is_same_v<T, const Register&>)
        {
          register_borrow<F, Args...>(self, fun, std::forward<Args>(args)...);
        }
        else if constexpr (std::is_same_v<T, Register&>)
        {
          register_ref<F, Args...>(self, fun, std::forward<Args>(args)...);
        }
        else if constexpr (std::is_same_v<T, Register>)
        {
          register_move<F, Args...>(self, fun, std::forward<Args>(args)...);
        }
        else if constexpr (std::is_base_of_v<
                             ConstantBase,
                             std::remove_reference_t<T>>)
        {
          load_constant_param<
            typename param_type_t<F, current_param>::T,
            F,
            Args...>(self, fun, std::forward<Args>(args)...);
        }
        else if constexpr (std::is_same_v<T, Thread&>)
        {
          process<F, Args..., Thread&>(
            self, fun, std::forward<Args>(args)..., self);
        }
        else if constexpr (std::is_same_v<T, Class&>)
        {
          load_class_param<F, Args...>(self, fun, std::forward<Args>(args)...);
        }
        else if constexpr (std::is_same_v<T, Frame&>)
        {
          process<F, Args..., Frame&>(
            self, fun, std::forward<Args>(args)..., *self.frame);
        }
        else if constexpr (std::is_same_v<T, Program&>)
        {
          process<F, Args..., Program&>(
            self, fun, std::forward<Args>(args)..., *self.program);
        }
        else if constexpr (std::is_same_v<T, Stack&>)
        {
          process<F, Args..., Stack&>(
            self, fun, std::forward<Args>(args)..., self.stack);
        }
        else if constexpr (std::is_same_v<T, ArgReg>)
        {
          self.trace_instruction(" ArgReg=", self.args);
          ArgReg arg{self.frame->arg(self.args++)};
          process<F, Args..., ArgReg>(
            self, fun, std::forward<Args>(args)..., arg);
        }
        else if constexpr (std::is_same_v<T, Function*>)
        {
          load_function_ptr_param<F, Args...>(
            self, fun, std::forward<Args>(args)...);
        }
        else
        {
          static_assert(
            always_false<T>::value,
            "Unsupported parameter type in Operands::process");
          abort();
        }
      }
    }
  };

  // Returns a cown containing the result of the behaviour.
  // The Value returns an rc to the cown.
  ValueTransfer Thread::run_async(uint32_t type_id, Function* func)
  {
    auto result = Cown::create(type_id);

    if (!Program::get().subtype(func->return_type, result->content_type_id()))
      Value::error(Error::BadType);

    auto construction =
      BehaviourCore::make(1, sizeof(Value) * 2, alignof(Value), run_behavior);
    BehaviourCore::initialise_request(
      construction,
      0,
      result,
      verona::rt::AccessMode::Write,
      verona::rt::Ownership::Borrowed);
    auto values = static_cast<Value*>(construction.body);
    new (&values[0]) Value(func);
    new (&values[1]) Value();
    BehaviourCore::schedule(BehaviourCore::finish_construction(construction));

    // Safe to convert to use Register this only contains a cown pointer, so
    // there is no requirement for a stack_rc.
    return ValueTransfer(result);
  }

  std::pair<Function*, PC> Thread::debug_info()
  {
    auto& t = get();

    if (t.frame)
      return {t.frame->func, t.current_pc};
    else
      return {t.behavior, t.current_pc};
  }

  Thread::Thread() : program(&Program::get()), frame(nullptr), args(0)
  {
    frames.reserve(16);
    locals.resize(1024);
  }

  Region* Thread::frame_region_for_stack(Location stack_loc)
  {
    assert(stack_loc.is_stack());
    auto& t = get();
    auto idx = stack_loc.stack_index();
    assert(idx < t.frames.size());
    return t.frames.at(idx).region;
  }

  Thread& Thread::get()
  {
    thread_local Thread thread;
    return thread;
  }

  void Thread::run_behavior(verona::rt::Work* work) noexcept
  {
    get().thread_run_behavior(work);
  }

  void Thread::handle_callback(CallbackClosure* cc, void* ret, void** args)
  {
    get().thread_handle_callback(cc, ret, args);
  }

  void Thread::thread_run_behavior(verona::rt::Work* work) noexcept
  {
    assert(!frame);
    assert(!args);
    auto b = BehaviourCore::from_work(work);
    auto values = b->get_body<Value>();
    behavior = values[0].function();
    auto& closure = values[1];

    if (!closure.is_invalid())
    {
      if (closure.is_header())
      {
        auto r = closure.get_header()->region();

        if (r && r->has_cown_owner())
        {
          // stack_inc to prevent auto-free during clear_cown_owner.
          // This becomes the register's stack ref — reg_dec on
          // teardown will balance it.
          r->stack_inc();
          r->clear_cown_owner();
        }
      }

      locals.at(args++) = ValueTransfer(closure);
    }

    // Populate cown arguments.
    auto num_cowns = b->get_count();
    auto result = b->acquired_cown(0);

    for (size_t i = 1; i < num_cowns; i++)
    {
      auto cown = b->acquired_cown(i);
      auto is_readonly = b->acquired_mode(i) == verona::rt::AccessMode::Read;

      // Writable cowns need an extra RC for the local register since
      // set_move will release the slot's RC separately.
      // Readonly cowns already carry an RC from the read instruction's
      // cown->inc() — don't add another or it will leak (the readonly
      // slot release path doesn't release the move RC with successors).
      if (!is_readonly)
        cown->inc();

      locals.at(args++) = ValueTransfer(cown, is_readonly);
    }

    try
    {
      // Store the function return value in the result cown.
      Register r = result->exchange<true>(thread_run(behavior));
    }
    catch (Value& error_value)
    {
      // Runtime error is fatal to the behavior. Clean up all frames and store
      // the error in the result cown.
      {
        logging::Error log;
        log << error_value.to_string() << std::endl;
        print_error_stack(log, error_value);
      }
      teardown_all();
      Register r = result->exchange<true>(ValueImmortal(error_value));
    }

#ifndef NDEBUG
    {
      logging::Log log(logging::detail::LogLevel::Trace);
      log << "Behavior complete. Final stack:" << std::endl;
      print_stack(log);
    }
#endif

    BehaviourCore::finished(work);
  }

  Register Thread::thread_run(Function* func)
  {
    auto depth = frames.size();
    Register saved;
    bool preserve_parent_local0 = frame != nullptr;

    if (preserve_parent_local0)
      saved = std::move(frame->local(0));

    try
    {
      pushframe(func, 0);

      invariant();

      while (depth != frames.size())
        step();
    }
    catch (Value&)
    {
      // An error may be raised before normal argument validation/consumption
      // resets `args` (for example while queueing a behavior). Clear any
      // pending arguments before unwinding so later sync calls don't trip the
      // `args == 0` invariant and mask the original error.
      drop_args();

      while (frames.size() > depth)
      {
        frame->drop_args(args);
        teardown();
        frames.pop_back();
        frame = frames.empty() ? nullptr : &frames.back();
      }

      if (preserve_parent_local0)
        frame->local(0) = std::move(saved);

      throw;
    }

    Register result;

    if (preserve_parent_local0)
    {
      result = std::move(frame->local(0));
      frame->local(0) = std::move(saved);
    }
    else
    {
      result = std::move(locals.at(0));
    }

    LOG(Trace) << "thread_run completed! Result: " << result;
    return result;
  }

  void Thread::print_stack(logging::Log& log, bool top_frame_only)
  {
    log << "Stack trace:" << std::endl;
    for (ssize_t i = frames.size() - 1; i >= 0; i--)
    {
      auto& frame = frames[i];
      log << "  at " << program->di_function(frame.func) << " (pc=" << frame.pc
          << ")";

      // print out the locals for this frame
      for (size_t j = 0; j < frame.func->registers; j++)
      {
        if (frame.local(j)->is_invalid())
          continue;
        log << std::endl << "    local[" << j << "] = " << frame.local(j);
      }

      if (top_frame_only)
        break;
      if (i != 0)
        log << std::endl;
    }

    if (args == 0)
      return;

    log << std::endl << "  Args stack:";
    for (size_t i = 0; i < args; i++)
    {
      log << std::endl << "    arg[" << i << "] = " << frame->arg(i);
    }
  }

  void Thread::print_error_stack(logging::Log& log, const Value& error_value)
  {
    auto error_func = error_value.error_function();
    auto error_pc = error_value.error_pc();

    log << "Stack trace:";

    ssize_t i = frames.size() - 1;

    if ((i >= 0) && (frames[i].func == error_func))
    {
      log << std::endl
          << "  at " << program->di_function(frames[i].func)
          << " (pc=" << error_pc << ")" << std::endl
          << program->debug_info(frames[i].func, error_pc);
      i--;
    }
    else
    {
      log << std::endl
          << "  at " << program->di_function(error_func) << " (pc=" << error_pc
          << ")" << std::endl
          << program->debug_info(error_func, error_pc);
    }

    for (; i >= 0; i--)
    {
      auto& frame = frames[i];
      auto start_pc = frame.func->labels.at(0);
      auto pc = frame.pc > start_pc ? frame.pc - 1 : frame.pc;

      log << std::endl
          << "  at " << program->di_function(frame.func) << " (pc=" << pc << ")"
          << std::endl
          << program->debug_info(frame.func, pc);
    }

    if (args == 0)
      return;

    log << std::endl << "  Args stack:";
    for (size_t i = 0; i < args; i++)
      log << std::endl << "    arg[" << i << "] = " << frame->arg(i);
  }

#ifndef NDEBUG
  void Thread::check_stack_rc_invariant(std::source_location loc)
  {
    std::unordered_map<Region*, size_t> calc_stack_rcs;

    auto visit_region = [&](Region* region) {
      // Walk parent chain for every region we have not already seen.
      // This accounts for the edge triggered stack RC from children to parents.
      while (calc_stack_rcs.find(region) == calc_stack_rcs.end() &&
             region->has_parent())
      {
        calc_stack_rcs[region] = 1;
        region = region->get_parent();
      }

      calc_stack_rcs[region] += 1;
    };

    auto visit_value = [&](ValueBorrow val) {
      // Readonly values do not contribute to stack rcs.
      if (val.is_readonly())
        return;

      auto loc = val.location();

      if (!loc.is_region())
        return;

      auto region = loc.to_region();

      // Frame-local regions skip stack RC accounting.
      if (region->is_frame_local())
        return;

      visit_region(region);
    };

    // For each register in the local frame, if it is a region reference, then
    for (auto& reg : locals)
    {
      visit_value(reg);
    }

    auto visit_header = [&](Header* h) {
      if (!h->location().is_region())
        return;

      auto r = h->location().to_region();

      if (!r->is_frame_local() && !r->has_cown_owner())
        visit_region(r);
    };

    for (auto& frame : frames)
    {
      frame.get_frame_local_region().trace_fn(visit_header);
    }

    stack.visit_headers({}, stack.top, [&](Header* h) {
      if (program->is_array(h->get_type_id()))
      {
        auto arr = static_cast<Array*>(h);
        arr->trace_fn(visit_header);
      }
      else
      {
        auto obj = static_cast<Object*>(h);
        obj->trace_fn(visit_header);
      }
    });

    // Now compare calculated stack RCs with actual stack RCs.
    for (const auto& [region, calc_stack_rc] : calc_stack_rcs)
    {
      auto actual_stack_rc = region->get_stack_rc();
      // Due to multi-threading it is not possible to guarantee a strict
      // equality between calculated and actual stack RCs. We would have to
      // pause all other threads to do that, which is not feasible. Instead we
      // check that the actual stack RC is at least the calculated stack RC.

      if (calc_stack_rc > actual_stack_rc)
      {
        logging::Error log;
        log << "Stack RC invariant violated at " << loc.file_name() << ":"
            << loc.line() << ":" << loc.column() << " for Region " << region
            << " calculated stack RC " << calc_stack_rc << " actual stack RC "
            << actual_stack_rc << std::endl;

        print_stack(log);
      }

      assert(calc_stack_rc <= actual_stack_rc && "Stack RC invariant violated");
    }
  }
#endif

  void Thread::invariant(std::source_location loc)
  {
#ifndef NDEBUG
    check_stack_rc_invariant(loc);
#else
    (void)loc;
#endif
  }

}

namespace vbc
{
  std::ostream& operator<<(std::ostream& os, Op op)
  {
    switch (op)
    {
      case Op::Const:
        return os << "Const";
      case Op::String:
        return os << "String";
      case Op::Convert:
        return os << "Convert";
      case Op::New:
        return os << "New";
      case Op::Stack:
        return os << "Stack";
      case Op::Heap:
        return os << "Heap";
      case Op::Region:
        return os << "Region";
      case Op::NewArray:
        return os << "NewArray";
      case Op::NewArrayConst:
        return os << "NewArrayConst";
      case Op::StackArray:
        return os << "StackArray";
      case Op::StackArrayConst:
        return os << "StackArrayConst";
      case Op::HeapArray:
        return os << "HeapArray";
      case Op::HeapArrayConst:
        return os << "HeapArrayConst";
      case Op::RegionArray:
        return os << "RegionArray";
      case Op::RegionArrayConst:
        return os << "RegionArrayConst";
      case Op::Copy:
        return os << "Copy";
      case Op::Move:
        return os << "Move";
      case Op::Drop:
        return os << "Drop";
      case Op::AtTeardown:
        return os << "AtTeardown";
      case Op::Freeze:
        return os << "Freeze";
      case Op::Pin:
        return os << "Pin";
      case Op::Unpin:
        return os << "Unpin";
      case Op::Merge:
        return os << "Merge";
      case Op::FFIStruct:
        return os << "FFIStruct";
      case Op::FFILoad:
        return os << "FFILoad";
      case Op::FFIStore:
        return os << "FFIStore";
      case Op::RegisterRef:
        return os << "RegisterRef";
      case Op::FieldRefMove:
        return os << "FieldRefMove";
      case Op::FieldRefCopy:
        return os << "FieldRefCopy";
      case Op::ArrayRefMove:
        return os << "ArrayRefMove";
      case Op::ArrayRefCopy:
        return os << "ArrayRefCopy";
      case Op::ArrayRefMoveConst:
        return os << "ArrayRefMoveConst";
      case Op::ArrayRefCopyConst:
        return os << "ArrayRefCopyConst";
      case Op::Load:
        return os << "Load";
      case Op::StoreMove:
        return os << "StoreMove";
      case Op::StoreCopy:
        return os << "StoreCopy";
      case Op::LookupDynamic:
        return os << "LookupDynamic";
      case Op::LookupFFI:
        return os << "LookupFFI";
      case Op::ArgMove:
        return os << "ArgMove";
      case Op::ArgCopy:
        return os << "ArgCopy";
      case Op::CallStatic:
        return os << "CallStatic";
      case Op::CallDynamic:
        return os << "CallDynamic";
      case Op::TryCallDynamic:
        return os << "TryCallDynamic";
      case Op::FFI:
        return os << "FFI";
      case Op::WhenStatic:
        return os << "WhenStatic";
      case Op::WhenDynamic:
        return os << "WhenDynamic";
      case Op::Typetest:
        return os << "Typetest";
      case Op::TailcallStatic:
        return os << "TailcallStatic";
      case Op::TailcallDynamic:
        return os << "TailcallDynamic";
      case Op::Return:
        return os << "Return";
      case Op::Raise:
        return os << "Raise";
      case Op::Cond:
        return os << "Cond";
      case Op::Jump:
        return os << "Jump";
      case Op::Add:
        return os << "Add";
      case Op::Sub:
        return os << "Sub";
      case Op::Mul:
        return os << "Mul";
      case Op::Div:
        return os << "Div";
      case Op::Mod:
        return os << "Mod";
      case Op::Pow:
        return os << "Pow";
      case Op::And:
        return os << "And";
      case Op::Or:
        return os << "Or";
      case Op::Xor:
        return os << "Xor";
      case Op::Shl:
        return os << "Shl";
      case Op::Shr:
        return os << "Shr";
      case Op::Eq:
        return os << "Eq";
      case Op::Ne:
        return os << "Ne";
      case Op::Lt:
        return os << "Lt";
      case Op::Le:
        return os << "Le";
      case Op::Gt:
        return os << "Gt";
      case Op::Ge:
        return os << "Ge";
      case Op::Min:
        return os << "Min";
      case Op::Max:
        return os << "Max";
      case Op::LogBase:
        return os << "LogBase";
      case Op::Atan2:
        return os << "Atan2";
      case Op::Neg:
        return os << "Neg";
      case Op::Not:
        return os << "Not";
      case Op::Abs:
        return os << "Abs";
      case Op::Ceil:
        return os << "Ceil";
      case Op::Floor:
        return os << "Floor";
      case Op::Exp:
        return os << "Exp";
      case Op::Log:
        return os << "Log";
      case Op::Sqrt:
        return os << "Sqrt";
      case Op::Cbrt:
        return os << "Cbrt";
      case Op::IsInf:
        return os << "IsInf";
      case Op::IsNaN:
        return os << "IsNaN";
      case Op::Sin:
        return os << "Sin";
      case Op::Cos:
        return os << "Cos";
      case Op::Tan:
        return os << "Tan";
      case Op::Asin:
        return os << "Asin";
      case Op::Acos:
        return os << "Acos";
      case Op::Atan:
        return os << "Atan";
      case Op::Sinh:
        return os << "Sinh";
      case Op::Cosh:
        return os << "Cosh";
      case Op::Tanh:
        return os << "Tanh";
      case Op::Asinh:
        return os << "Asinh";
      case Op::Acosh:
        return os << "Acosh";
      case Op::Atanh:
        return os << "Atanh";
      case Op::Bits:
        return os << "Bits";
      case Op::Len:
        return os << "Len";
      case Op::Ptr:
        return os << "Ptr";
      case Op::Read:
        return os << "Read";
      case Op::GetRaise:
        return os << "GetRaise";
      case Op::SetRaise:
        return os << "SetRaise";
      case Op::Const_E:
        return os << "Const_E";
      case Op::Const_Pi:
        return os << "Const_Pi";
      case Op::Const_Inf:
        return os << "Const_Inf";
      case Op::Const_NaN:
        return os << "Const_NaN";
      case Op::MakeCallback:
        return os << "MakeCallback";
      case Op::CodePtrCallback:
        return os << "CodePtrCallback";
      case Op::FreeCallback:
        return os << "FreeCallback";
      case Op::AddExternal:
        return os << "AddExternal";
      case Op::RemoveExternal:
        return os << "RemoveExternal";
      case Op::MemoLoad:
        return os << "MemoLoad";
      case Op::ArrayCopy:
        return os << "ArrayCopy";
      case Op::ArrayFill:
        return os << "ArrayFill";
      case Op::ArrayCompare:
        return os << "ArrayCompare";
      default:
        return os << "Unknown";
    }
  }
}

namespace vbci
{
  void Thread::step()
  {
    assert(frame);
    current_pc = frame->pc;
    auto op = leb<Op>();
    auto process =
      [this](auto f, std::source_location loc = std::source_location::current())
        INLINE {
          Operands::process(*this, f);
          // Check the invariant after running this instruction.
          invariant(loc);
        };

    trace_instruction("OP:", op);
    switch (op)
    {
      case Op::Const:
      {
        process([](Thread& self, Register& dst, Constant<PrimitiveType> t)
                  INLINE {
                    switch (t)
                    {
                      case PrimitiveType::None:
                        dst = ValueImmortal(Value::none());
                        break;

                      case PrimitiveType::Bool:
                      {
                        auto value = self.leb<bool>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::I8:
                      {
                        auto value = self.leb<int8_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::I16:
                      {
                        auto value = self.leb<int16_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::I32:
                      {
                        auto value = self.leb<int32_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::I64:
                      {
                        auto value = self.leb<int64_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::U8:
                      {
                        auto value = self.leb<uint8_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::U16:
                      {
                        auto value = self.leb<uint16_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::U32:
                      {
                        auto value = self.leb<uint32_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::U64:
                      {
                        auto value = self.leb<uint64_t>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::ILong:
                      {
                        auto value = self.leb<int64_t>();
                        dst = Value::from_ffi(value_type(t), value);
                        break;
                      }
                      case PrimitiveType::ISize:
                      {
                        auto value = self.leb<int64_t>();
                        dst = Value::from_ffi(value_type(t), value);
                        break;
                      }

                      case PrimitiveType::ULong:
                      {
                        auto value = self.leb<uint64_t>();
                        dst = Value::from_ffi(value_type(t), value);
                        break;
                      }

                      case PrimitiveType::USize:
                      {
                        auto value = self.leb<uint64_t>();
                        dst = Value::from_ffi(value_type(t), value);
                        break;
                      }

                      case PrimitiveType::F32:
                      {
                        auto value = self.leb<float>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      case PrimitiveType::F64:
                      {
                        auto value = self.leb<double>();
                        dst = ValueImmortal(value);
                        break;
                      }

                      default:
                        Value::error(Error::BadConversion);
                    }
                  });
        break;
      }

      case Op::String:
      {
        process(
          [](Program& program, Register& dst, Constant<size_t> string_id)
            INLINE { dst = ValueImmortal(program.get_string(string_id)); });
        break;
      }

      case Op::Convert:
      {
        process(
          [](Register& dst, Constant<PrimitiveType> t, const Register& src)
            INLINE { dst = ValueImmortal(src->convert(value_type(t))); });
        break;
      }

      case Op::New:
      {
        process(
          [](Register& dst, Class& cls, Thread& self, Frame& frame) INLINE {
            self.check_args(cls.fields);
            dst = ValueTransfer(&frame.region->object(cls)->init(frame, cls));
          });
        break;
      }

      case Op::Stack:
      {
        process(
          [](
            Register& dst, Class& cls, Thread& self, Frame& frame, Stack& stack)
            INLINE {
              self.check_args(cls.fields);
              auto mem = stack.alloc(cls.size);
              auto obj =
                &Object::create(mem, cls, frame.frame_id)->init(frame, cls);
              frame.push_finalizer(obj);
              dst = ValueTransfer(obj);
            });
        break;
      }

      case Op::Heap:
      {
        process([](
                  Register& dst,
                  const Register& region_loc,
                  Class& cls,
                  Thread& self,
                  Frame& frame) INLINE {
          auto region = region_loc->region();
          self.check_args(cls.fields);
          dst = ValueTransfer(&region->object(cls)->init(frame, cls));
        });
        break;
      }

      case Op::Region:
      {
        process([](
                  Register& dst,
                  Constant<RegionType> region_type,
                  Class& cls,
                  Thread& self,
                  Frame& frame) INLINE {
          self.check_args(cls.fields);
          auto region = Region::create(region_type);
          dst = ValueTransfer(&region->object(cls)->init(frame, cls));
        });
        break;
      }

      case Op::NewArray:
      {
        process([](
                  Register& dst,
                  const Register& size,
                  Constant<size_t> type_id,
                  Frame& frame) INLINE {
          dst = ValueTransfer(frame.region->array(type_id, size->get_size()));
        });
        break;
      }

      case Op::NewArrayConst:
      {
        process([](
                  Register& dst,
                  Constant<size_t> type_id,
                  Constant<size_t> size,
                  Frame& frame) INLINE {
          dst = ValueTransfer(frame.region->array(type_id, size));
        });
        break;
      }

      case Op::StackArray:
      {
        process([](
                  Register& dst,
                  Stack& stack,
                  const Register& size,
                  Constant<size_t> type_id,
                  Frame& frame) INLINE {
          dst = ValueTransfer(
            stack.array(frame.frame_id, type_id, size->get_size()));
        });
        break;
      }

      case Op::StackArrayConst:
      {
        process([](
                  Register& dst,
                  Constant<size_t> type_id,
                  Constant<size_t> size,
                  Stack& stack,
                  Frame& frame) INLINE {
          dst = ValueTransfer(stack.array(frame.frame_id, type_id, size));
        });
        break;
      }

      case Op::HeapArray:
      {
        process([](
                  Register& dst,
                  const Register& region_loc,
                  const Register& size,
                  Constant<size_t> type_id) INLINE {
          auto region = region_loc->region();
          dst = ValueTransfer(region->array(type_id, size->get_size()));
        });
        break;
      }

      case Op::HeapArrayConst:
      {
        process([](
                  Register& dst,
                  const Register& region_loc,
                  Constant<size_t> type_id,
                  Constant<size_t> size) INLINE {
          auto region = region_loc->region();
          dst = ValueTransfer(region->array(type_id, size));
        });
        break;
      }

      case Op::RegionArray:
      {
        process([](
                  Register& dst,
                  Constant<RegionType> region_type,
                  const Register& size,
                  Constant<size_t> type_id) INLINE {
          auto region = Region::create(region_type);
          dst = ValueTransfer(region->array(type_id, size->get_size()));
        });
        break;
      }

      case Op::RegionArrayConst:
      {
        process([](
                  Register& dst,
                  Constant<RegionType> region_type,
                  Constant<size_t> type_id,
                  Constant<size_t> size) INLINE {
          auto region = Region::create(region_type);
          dst = ValueTransfer(region->array(type_id, size));
        });
        break;
      }

      case Op::Copy:
      {
        process([](Register& dst, const Register& src) INLINE { dst = src; });
        break;
      }

      case Op::Move:
      {
        process([](Register& dst, Register src)
                  INLINE { dst = std::move(src); });
        break;
      }

      case Op::Drop:
      {
        process([](Register) INLINE {});
        break;
      }

      case Op::AtTeardown:
      {
        process([](Register callback) INLINE {
          if (callback->type() == ValueType::None)
            return;

          auto* apply = callback->method(CallbackMethodId);
          if (!apply)
            Value::error(Error::MethodNotFound);

          auto location = callback->location();
          if (location.is_stack())
            Value::error(Error::BadStackEscape);

          if (location.is_region() && location.to_region()->is_frame_local())
          {
            auto* region = Region::create(RegionType::RegionRC);
            if (!drag_allocation<false>(region, callback->get_header()))
            {
              region->free_region();
              Value::error(Error::BadStackEscape);
            }
          }

          auto* work = verona::rt::Closure::make(
            [callback = std::move(callback), apply](verona::rt::Work*) mutable {
              Thread::run_cleanup(apply, callback.borrow());
              return true;
            });
          verona::rt::Scheduler::schedule_at_quiescence(work);
        });
        break;
      }

      case Op::Freeze:
      {
        process([](Register& dst, Register& src) INLINE {
          if (src->is_readonly())
            Value::error(Error::BadFreeze);

          if (src->is_header())
          {
            if (!freeze(src->get_header()))
              Value::error(Error::BadFreeze);
          }

          Register value = std::move(src);
          dst = std::move(value);
        });
        break;
      }

      case Op::RegisterRef:
      {
        process([](Register& dst, Constant<size_t> idx, Frame& frame) INLINE {
          auto params = frame.func->param_types.size();
          auto vars = frame.func->var_types.size();

          if (idx < params || idx >= params + vars)
            Value::error(Error::BadRefTarget);

          dst = ValueStaticLifetime(frame.base + idx, frame.frame_id);
        });
        break;
      }

      case Op::FieldRefMove:
      {
        process(
          [](Register& dst, Register src, Constant<size_t> field_id)
            INLINE { dst.from_field_ref<true>(std::move(src), field_id); });
        break;
      }

      case Op::FieldRefCopy:
      {
        process([](Register& dst, Register& src, Constant<size_t> field_id)
                  INLINE { dst.from_field_ref<false>(src, field_id); });
        break;
      }

      case Op::ArrayRefMove:
      {
        process([](Register& dst, Register src, const Register& idx) INLINE {
          dst.from_array_ref<true>(std::move(src), idx->get_size());
        });
        break;
      }

      case Op::ArrayRefCopy:
      {
        process([](Register& dst, const Register& src, const Register& idx)
                  INLINE { dst.from_array_ref<false>(src, idx->get_size()); });
        break;
      }

      case Op::ArrayRefMoveConst:
      {
        process([](Register& dst, Register src, Constant<size_t> idx)
                  INLINE { dst.from_array_ref<true>(std::move(src), idx); });
        break;
      }

      case Op::ArrayRefCopyConst:
      {
        process([](Register& dst, const Register& src, Constant<size_t> idx)
                  INLINE { dst.from_array_ref<false>(src, idx); });
        break;
      }

      case Op::Load:
      {
        process([](Register& dst, const Register& src)
                  INLINE { dst.from_load(src); });
        break;
      }

      case Op::StoreMove:
      {
        process([](Register& dst, const Register& ref, Register src)
                  INLINE { dst.from_exchange<true>(ref, std::move(src)); });
        break;
      }

      case Op::StoreCopy:
      {
        process([](Register& dst, const Register& ref, const Register& src)
                  INLINE { dst.from_exchange<false>(ref, src); });
        break;
      }

      case Op::LookupDynamic:
      {
        process(
          [](Register& dst, const Register& src, Constant<size_t> method_id)
            INLINE {
              auto f = src->method(method_id);

              if (!f)
                dst = ValueImmortal(Value());
              else
                dst = ValueImmortal(f);
            });
        break;
      }

      case Op::LookupFFI:
      {
        process([](Register& dst, Constant<size_t> symbol_id, Program& program)
                  INLINE {
                    dst =
                      ValueImmortal(program.symbol(symbol_id).raw_pointer());
                  });
        break;
      }

      case Op::ArgMove:
      {
        process([](Register src, ArgReg dst) INLINE { dst = std::move(src); });
        break;
      }

      case Op::ArgCopy:
      {
        process([](const Register& src, ArgReg dst) INLINE { dst = src; });
        break;
      }

      case Op::CallStatic:
      {
        process([](Constant<size_t> dst_id, Function* func, Thread& self)
                  INLINE { self.pushframe(func, dst_id); });
        break;
      }

      case Op::CallDynamic:
      {
        process([](Constant<size_t> dst_id, const Register& func, Thread& self)
                  INLINE { self.pushframe(func->function(), dst_id); });
        break;
      }

      case Op::TryCallDynamic:
      {
        process([](Constant<size_t> dst_id, const Register& func, Thread& self)
                  INLINE { self.try_pushframe(func->function(), dst_id); });
        break;
      }

      case Op::FFI:
      {
        process([](
                  Register& dst,
                  Constant<size_t> symbol_id,
                  Thread& self,
                  Frame& frame,
                  Program& program) INLINE {
          auto num_args = self.args;
          auto& symbol = program.symbol(symbol_id);
          auto& params = symbol.params();
          auto& paramvals = symbol.paramvals();

          if (
            (num_args < params.size()) ||
            (!symbol.varargs() && (num_args > params.size())))
          {
            self.drop_args();
            Value::error(Error::BadArgs);
          }

          for (size_t i = 0; i < params.size(); i++)
          {
            auto& arg = frame.arg(i);

            if (params.at(i) == +ValueType::Ptr)
            {
              if (!ffi_ptr_compatible(arg))
              {
                self.drop_args();
                Value::error(Error::BadType);
              }
            }
            else if (!program.subtype(arg->type_id(), params.at(i)))
            {
              self.drop_args();
              Value::error(Error::BadType);
            }
          }

          self.args = 0;

          // A Value must be passed as a pointer, not as a struct, since it
          // is a C++ non-trivally constructed type.

          auto& ffi_arg_addrs = self.ffi_arg_addrs;
          auto& ffi_arg_vals = self.ffi_arg_vals;

          if (ffi_arg_addrs.size() < num_args)
          {
            ffi_arg_addrs.resize(num_args);
            ffi_arg_vals.resize(num_args);
          }

          for (size_t i = 0; i < num_args; i++)
          {
            auto& arg = frame.arg(i);
            ValueType vt;

            if (i < params.size())
            {
              vt = paramvals.at(i);
            }
            else
            {
              auto rep = program.layout_type_id(arg->type_id());
              // See Symbol::prepare: `Value`s cross the boundary as pointers,
              // so the CIF must declare pointer-sized args, not the 16-byte
              // by-value struct that layout_type_id returns for `Dyn`.
              auto* cif_type =
                (rep.first == ValueType::Dyn) ? &ffi_type_pointer : rep.second;
              symbol.varparam(cif_type);
              vt = rep.first;
            }

            if (vt == ValueType::Dyn)
            {
              // Dynamic type: pass a pointer to the Value.
              ffi_arg_vals.at(i) = &arg;
              ffi_arg_addrs.at(i) = &ffi_arg_vals.at(i);
            }
            else if (vt == ValueType::Ptr)
            {
              marshal_ffi_ptr_arg(arg, ffi_arg_addrs.at(i), ffi_arg_vals.at(i));
            }
            else if (vt == ValueType::Object)
            {
              // Object: pass a pointer to the fields (past the header).
              ffi_arg_vals.at(i) = arg->get_object()->get_pointer();
              ffi_arg_addrs.at(i) = &ffi_arg_vals.at(i);
            }
            else if (vt == ValueType::Array)
            {
              // Array: pass a pointer to the elements (past the header).
              ffi_arg_vals.at(i) = arg->get_array()->get_pointer();
              ffi_arg_addrs.at(i) = &ffi_arg_vals.at(i);
            }
            else
            {
              ffi_arg_addrs.at(i) = arg->to_ffi();
            }
          }

          auto ret = symbol.call(ffi_arg_addrs);

          if (!ret.is_error() && !program.subtype(ret.type_id(), symbol.ret()))
            Value::error(Error::BadType);

          // TODO: Is the FFI guaranteed to return a value that has an RC?
          dst = ValueTransfer(ret);
          frame.drop_args(num_args);
        });
        break;
      }

      case Op::WhenStatic:
      {
        process([](
                  Register& dst,
                  Constant<size_t> type_id,
                  Constant<size_t> func_id,
                  Thread& self,
                  Program& program) INLINE {
          auto func = program.function(func_id);
          self.queue_behavior(dst, type_id, func);
        });
        break;
      }

      case Op::WhenDynamic:
      {
        process([](
                  Register& dst,
                  Constant<size_t> type_id,
                  const Register& func,
                  Thread& self) INLINE {
          self.queue_behavior(dst, type_id, func->function());
        });
        break;
      }

      case Op::Typetest:
      {
        process([](
                  Register& dst,
                  const Register& src,
                  Constant<size_t> type_id,
                  Program& program) INLINE {
          dst = ValueImmortal(Value(program.subtype(src->type_id(), type_id)));
        });
        break;
      }

      case Op::TailcallStatic:
      {
        process([](Constant<size_t> func_id, Program& program, Thread& self)
                  INLINE { self.tailcall(program.function(func_id)); });
        break;
      }

      case Op::TailcallDynamic:
      {
        process([](const Register& func, Thread& self)
                  INLINE { self.tailcall(func->function()); });
        break;
      }

      case Op::Return:
      {
        process([](Register ret, Thread& self)
                  INLINE { self.popframe(std::move(ret)); });
        break;
      }

      case Op::Raise:
      {
        process([](Register ret, Thread& self) INLINE {
          self.raise(std::move(ret), self.frame->raise_target);
        });
        break;
      }

      case Op::Cond:
      {
        process([](
                  const Register& cond,
                  Constant<size_t> on_true,
                  Constant<size_t> on_false,
                  Thread& self) INLINE {
          if (cond->get_bool())
            self.branch(on_true);
          else
            self.branch(on_false);
        });
        break;
      }

      case Op::Jump:
      {
        process([](Constant<size_t> target, Thread& self)
                  INLINE { self.branch(target); });
        break;
      }

#define do_binop(opname) \
  { \
    process( \
      [](Register& dst, const Register& lhs, const Register& rhs) \
        INLINE { dst = ValueImmortal(lhs->op_##opname(rhs.borrow())); }); \
    break; \
  }
      case Op::Add:
        do_binop(add);
      case Op::Sub:
        do_binop(sub);
      case Op::Mul:
        do_binop(mul);
      case Op::Div:
        do_binop(div);
      case Op::Mod:
        do_binop(mod);
      case Op::Pow:
        do_binop(pow);
      case Op::And:
        do_binop(and);
      case Op::Or:
        do_binop(or);
      case Op::Xor:
        do_binop(xor);
      case Op::Shl:
        do_binop(shl);
      case Op::Shr:
        do_binop(shr);
      case Op::Eq:
        do_binop(eq);
      case Op::Ne:
        do_binop(ne);
      case Op::Lt:
        do_binop(lt);
      case Op::Le:
        do_binop(le);
      case Op::Gt:
        do_binop(gt);
      case Op::Ge:
        do_binop(ge);
      case Op::Min:
        do_binop(min);
      case Op::Max:
        do_binop(max);
      case Op::LogBase:
        do_binop(logbase);
      case Op::Atan2:
        do_binop(atan2);

#define do_unop(opname) \
  { \
    process([](Register& dst, const Register& src) \
              INLINE { dst = ValueImmortal(src->op_##opname()); }); \
    break; \
  }
      case Op::Neg:
        do_unop(neg);
      case Op::Not:
        do_unop(not);
      case Op::Abs:
        do_unop(abs);
      case Op::Ceil:
        do_unop(ceil);
      case Op::Floor:
        do_unop(floor);
      case Op::Exp:
        do_unop(exp);
      case Op::Log:
        do_unop(log);
      case Op::Sqrt:
        do_unop(sqrt);
      case Op::Cbrt:
        do_unop(cbrt);
      case Op::IsInf:
        do_unop(isinf);
      case Op::IsNaN:
        do_unop(isnan);
      case Op::Sin:
        do_unop(sin);
      case Op::Cos:
        do_unop(cos);
      case Op::Tan:
        do_unop(tan);
      case Op::Asin:
        do_unop(asin);
      case Op::Acos:
        do_unop(acos);
      case Op::Atan:
        do_unop(atan);
      case Op::Sinh:
        do_unop(sinh);
      case Op::Cosh:
        do_unop(cosh);
      case Op::Tanh:
        do_unop(tanh);
      case Op::Asinh:
        do_unop(asinh);
      case Op::Acosh:
        do_unop(acosh);
      case Op::Atanh:
        do_unop(atanh);
      case Op::Bits:
        do_unop(bits);
      case Op::Len:
        do_unop(len);
      case Op::Read:
      {
        process([](Register& dst, const Register& src)
                  INLINE { dst = src->op_read(); });
        break;
      }

      case Op::Ptr:
        // TODO unsure about this one, does it really make sense?
        {
          process([](Register& dst, Register& src) INLINE {
            // This feels insanely dangerous, and needs review.
            dst = ValueImmortal(src.borrow().op_ptr());
          });
          break;
        }

      case Op::GetRaise:
      {
        process([](Register& dst, Frame& frame) INLINE {
          dst = ValueImmortal(
            Value(static_cast<uint64_t>(frame.raise_target.raw())));
        });
        break;
      }

      case Op::SetRaise:
      {
        process([](Register& dst, const Register& src, Frame& frame) INLINE {
          dst = ValueImmortal(
            Value(static_cast<uint64_t>(frame.raise_target.raw())));
          frame.raise_target = Location::from_raw(src->get_u64());
        });
        break;
      }

#define do_const(opname) \
  { \
    process([](Register& dst) \
              INLINE { dst = ValueImmortal(Value::opname()); }); \
    break; \
  }
      case Op::Const_E:
        do_const(e);
      case Op::Const_Pi:
        do_const(pi);
      case Op::Const_Inf:
        do_const(inf);
      case Op::Const_NaN:
        do_const(nan);

      case Op::MakeCallback:
      {
        process([](Register& dst, const Register& src) INLINE {
          auto* func = src->method(CallbackMethodId);

          if (!func)
            Value::error(Error::MethodNotFound);

          auto cc = make_callback(src, func);
          dst = ValueTransfer(Value(cc));
        });
        break;
      }

      case Op::CodePtrCallback:
      {
        process([](Register& dst, const Register& src) INLINE {
          auto* cc = static_cast<CallbackClosure*>(src->get_ptr());

          if (!cc)
            Value::error(Error::BadOperand);

          dst = ValueTransfer(Value(cc->code_ptr));
        });
        break;
      }

      case Op::FreeCallback:
      {
        process([](Register& dst, const Register& src) INLINE {
          auto* cc = static_cast<CallbackClosure*>(src->get_ptr());

          if (!cc)
            Value::error(Error::BadOperand);

          cc->free();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::Pin:
      {
        process([](Register& dst, const Register& src) INLINE {
          src->pin();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::Unpin:
      {
        process([](Register& dst, const Register& src) INLINE {
          src->unpin();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::Merge:
      {
        process([](Register& dst, const Register& a, const Register& b) INLINE {
          merge(a, b);
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::FFIStruct:
      {
        process([](
                  Register& dst,
                  Constant<size_t> layout_type_id,
                  Frame& frame,
                  Program& program) INLINE {
          std::vector<ValueType> kinds;
          std::vector<ffi_type*> reps;

          if (!ffi_struct_members(program, layout_type_id, kinds, reps))
            Value::error(Error::BadType);

          std::vector<ffi_type*> struct_elements = reps;
          struct_elements.push_back(nullptr);

          std::vector<size_t> offsets(kinds.size());
          ffi_type struct_type;
          struct_type.size = 0;
          struct_type.alignment = 0;
          struct_type.type = FFI_TYPE_STRUCT;
          struct_type.elements = struct_elements.data();

          if (
            ffi_get_struct_offsets(
              FFI_DEFAULT_ABI, &struct_type, offsets.data()) != FFI_OK)
          {
            Value::error(Error::BadType);
          }

          auto result_type_id = program.get_typeid_ffi_struct_result();
          auto offsets_type_id = program.get_typeid_array_usize();
          auto kinds_type_id = program.get_typeid_arg();

          auto* offsets_arr =
            frame.region->array(offsets_type_id, offsets.size());
          auto* kinds_arr = frame.region->array(kinds_type_id, kinds.size());
          auto* tuple_arr = frame.region->array(result_type_id, 3);

          for (size_t i = 0; i < offsets.size(); i++)
          {
            Register off_reg =
              ValueTransfer(Value(ValueType::USize, offsets.at(i)));
            Register kind_reg = ValueTransfer(
              Value(ValueType::U8, static_cast<uint8_t>(kinds.at(i))));
            offsets_arr->exchange<false>(i, off_reg);
            kinds_arr->exchange<false>(i, kind_reg);
          }

          Register size_reg =
            ValueTransfer(Value(ValueType::USize, struct_type.size));
          Register offsets_reg = ValueTransfer(offsets_arr);
          Register kinds_reg = ValueTransfer(kinds_arr);

          tuple_arr->exchange<false>(0, size_reg);
          tuple_arr->exchange<false>(1, offsets_reg);
          tuple_arr->exchange<false>(2, kinds_reg);

          if (!freeze(tuple_arr))
            Value::error(Error::BadType);

          dst = ValueTransfer(tuple_arr);
        });
        break;
      }

      case Op::FFILoad:
      {
        process([](
                  Register& dst,
                  const Register& base,
                  const Register& offset,
                  const Register& kind_reg,
                  Constant<size_t> expected_type_id,
                  Program& program) INLINE {
          auto actual_kind = static_cast<ValueType>(kind_reg->get_u8());
          auto expected_kind = ffi_expected_kind(program, expected_type_id);

          if (actual_kind != expected_kind)
            Value::error(Error::BadType);

          auto value =
            Value::from_addr(actual_kind, ffi_field_addr(base, offset));

          if (
            (value.is_header() || value.is_cown()) &&
            !program.subtype(value.type_id(), expected_type_id))
          {
            Value::error(Error::BadType);
          }

          dst = value;
        });
        break;
      }

      case Op::FFIStore:
      {
        process([](
                  Register& dst,
                  const Register& base,
                  const Register& offset,
                  const Register& kind_reg,
                  const Register& value,
                  Constant<size_t> expected_type_id,
                  Program& program) INLINE {
          auto actual_kind = static_cast<ValueType>(kind_reg->get_u8());
          auto expected_kind = ffi_expected_kind(program, expected_type_id);

          if (actual_kind != expected_kind)
            Value::error(Error::BadType);

          if (!program.subtype(value->type_id(), expected_type_id))
            Value::error(Error::BadType);

          value->to_addr(actual_kind, ffi_field_addr(base, offset));
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::AddExternal:
      {
        process([](Register& dst) INLINE {
          verona::rt::Scheduler::add_external_event_source();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::RemoveExternal:
      {
        process([](Register& dst) INLINE {
          verona::rt::Scheduler::remove_external_event_source();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::MemoLoad:
      {
        process([](Register& dst, Constant<size_t> slot) INLINE {
          auto& val = Program::get().memo_slot(slot);
          dst = val.borrow();
        });
        break;
      }

      case Op::ArrayCopy:
      {
        process([](Register& dst, Thread& self, Frame& frame) INLINE {
          auto& dst_arr_reg = frame.arg(0);
          auto& dst_off_reg = frame.arg(1);
          auto& src_arr_reg = frame.arg(2);
          auto& src_off_reg = frame.arg(3);
          auto& len_reg = frame.arg(4);

          auto* dst_arr = dst_arr_reg->get_array();
          auto* src_arr = src_arr_reg->get_array();
          auto dst_off = dst_off_reg->get_size();
          auto src_off = src_off_reg->get_size();
          auto len = len_reg->get_size();

          dst_arr->bulk_copy(dst_off, src_arr, src_off, len);
          self.drop_args();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::ArrayFill:
      {
        process([](Register& dst, Thread& self, Frame& frame) INLINE {
          auto& dst_arr_reg = frame.arg(0);
          auto& off_reg = frame.arg(1);
          auto& len_reg = frame.arg(2);
          auto& val_reg = frame.arg(3);

          auto* dst_arr = dst_arr_reg->get_array();
          auto off = off_reg->get_size();
          auto len = len_reg->get_size();

          dst_arr->bulk_fill(
            off, len, static_cast<const Value&>(val_reg.borrow()));
          self.drop_args();
          dst = ValueImmortal(Value::none());
        });
        break;
      }

      case Op::ArrayCompare:
      {
        process([](Register& dst, Thread& self, Frame& frame) INLINE {
          auto& a_reg = frame.arg(0);
          auto& a_off_reg = frame.arg(1);
          auto& b_reg = frame.arg(2);
          auto& b_off_reg = frame.arg(3);
          auto& len_reg = frame.arg(4);

          auto* a = a_reg->get_array();
          auto* b = b_reg->get_array();
          auto a_off = a_off_reg->get_size();
          auto b_off = b_off_reg->get_size();
          auto len = len_reg->get_size();

          auto result = a->bulk_compare(a_off, b, b_off, len);
          self.drop_args();
          dst = ValueImmortal(Value(static_cast<int64_t>(result)));
        });
        break;
      }

      default:
        Value::error(Error::UnknownOpcode);
    }
  }

  void Thread::pushframe(Function* func, size_t dst)
  {
    if (!func)
      Value::error(Error::MethodNotFound);

    LOG(Trace) << "Call " << program->di_function(func);

    check_args(func->param_types);

    Location frame_id = Location::stack();
    size_t base = 0;
    size_t finalize_base = 0;

    if (frame)
    {
      frame_id = frame->frame_id.next_stack_level();
      base = frame->base + frame->func->registers;
      finalize_base = frame->finalize_top;
    }

    // Make sure there's enough register space.
    auto req_stack_size = base + func->registers;

    while (locals.size() < req_stack_size)
      locals.resize(locals.size() * 2);

    frames.emplace_back(
      func,
      frame_id,
      stack.save(),
      locals,
      base,
      finalize,
      finalize_base,
      func->labels.at(0),
      dst);

    frame = &frames.back();
  }

  void Thread::try_pushframe(Function* func, size_t dst)
  {
    if (!func)
    {
      drop_args();
      frame->local(dst) = ValueImmortal(Value());
      return;
    }

    if (!try_check_args(func->param_types))
    {
      // try_check_args already dropped args on failure.
      frame->local(dst) = ValueImmortal(Value());
      return;
    }

    LOG(Trace) << "TryCall " << program->di_function(func);

    Location frame_id = Location::stack();
    size_t base = 0;
    size_t finalize_base = 0;

    if (frame)
    {
      frame_id = frame->frame_id.next_stack_level();
      base = frame->base + frame->func->registers;
      finalize_base = frame->finalize_top;
    }

    // Make sure there's enough register space.
    auto req_stack_size = base + func->registers;

    while (locals.size() < req_stack_size)
      locals.resize(locals.size() * 2);

    frames.emplace_back(
      func,
      frame_id,
      stack.save(),
      locals,
      base,
      finalize,
      finalize_base,
      func->labels.at(0),
      dst);

    frame = &frames.back();
  }

  void Thread::popframe(Register ret_)
  {
    // Create temp register to keep return alive across dropping of the frame.
    Register ret{std::move(ret_)};

    // Save the destination register.
    auto dst = frame->dst;

    // Clear any unused arguments.
    frame->drop_args(args);

    // Check for stack escapes.
    auto retloc = ret->location();

    if (retloc == frame->frame_id)
    {
      // The return value can't be stack allocated in this frame.
      Value::error(Error::BadStackEscape);
    }
    else if (retloc.is_region() && retloc.to_region() == frame->region)
    {
      if (frames.size() > 1)
      {
        // Drag the frame-local allocation to the previous frame's region.
        auto& prev_frame = frames.at(frames.size() - 2);

        if (!drag_allocation<false>(prev_frame.region, ret->get_base_header()))
          Value::error(Error::BadStackEscape);
      }
      else
      {
        // Drag the frame-local allocation to a fresh region.
        auto r = Region::create(RegionType::RegionRC);
        // This is a fresh region, so we need the stack rc from the entry, so do
        // not count this as a move.
        if (!drag_allocation<false>(r, ret->get_base_header()))
        {
          r->free_region();
          Value::error(Error::BadStackEscape);
        }
      }
    }

    // Check the return type.
    if (
      !ret->is_error() &&
      !program->subtype(ret->type_id(), frame->func->return_type))
    {
      Value::error(Error::BadType);
    }

    teardown();
    frames.pop_back();

    if (frames.empty())
    {
      locals.at(0) = std::move(ret);
      frame = nullptr;
      return;
    }

    frame = &frames.back();
    frame->local(dst) = std::move(ret);
  }

  void Thread::raise(Register ret_, Location target)
  {
    Register ret{std::move(ret_)};

    // Validate the target: must be a stack frame with a lower ID than the
    // current frame (tailcalls retain frame IDs, so no gaps exist).
    if (!target.is_stack() || target >= frame->frame_id)
      Value::error(Error::BadRaiseTarget);

    // Check if the raised value needs relocation before we tear down
    // intermediate frames.
    auto retloc = ret->location();

    if (retloc.is_stack() && retloc > target)
    {
      // Stack-allocated in a frame younger than the target. It will be
      // destroyed when we tear down intermediate frames.
      Value::error(Error::BadStackEscape);
    }

    if (
      retloc.is_region() && retloc.to_region()->is_frame_local() &&
      (retloc.to_region()->get_frame_depth() > (target.stack_index() + 1)))
    {
      // Frame-local in a frame younger than the target. Drag it to the
      // target frame's region so it survives intermediate teardowns.
      auto target_region = frame_region_for_stack(target);

      if (!drag_allocation<false>(target_region, ret->get_base_header()))
        Value::error(Error::BadStackEscape);
    }

    // Tear down all intermediate frames.
    while (frame->frame_id != target)
    {
      frame->drop_args(args);
      teardown();
      frames.pop_back();
      frame = &frames.back();
    }

    // Target frame reached. Convert to a normal return.
    popframe(std::move(ret));
  }

  void Thread::tailcall(Function* func)
  {
    if (!func)
      Value::error(Error::MethodNotFound);

    // Preserve the frame-local region.
    teardown(true);
    check_args(func->param_types);

    // Move arguments back to the current frame.
    bool stack_escape = false;

    for (size_t i = 0; i < func->param_types.size(); i++)
    {
      auto& arg = frame->arg(i);

      if (arg->location() == frame->frame_id)
        stack_escape = true;

      frame->local(i) = std::move(arg);
    }

    // Can't tailcall with stack allocations.
    if (stack_escape)
      Value::error(Error::BadStackEscape);

    // Set the new function and program counter.
    frame->func = func;
    frame->pc = func->labels.at(0);
  }

  void Thread::teardown(bool tailcall)
  {
    LOG(Trace) << "Tearing down frame " << program->di_function(frame->func);
    // Drop all frame registers.
    frame->drop();

    // Finalize all stack objects (run finalizers and drop field references).
    stack.visit_headers(frame->save, stack.top, [&](Header* h) {
      if (Program::get().is_array(h->get_type_id()))
        static_cast<Array*>(h)->finalize();
      else
        static_cast<Object*>(h)->finalize();
    });

    // Finalize the frame-local region. A tailcall preserves the region.
    if (!tailcall)
    {
      assert(frame->region->is_frame_local());
      collect(frame->region);
    }

    // Pop the stack.
    stack.restore(frame->save);
  }

  void Thread::teardown_all()
  {
    while (frame)
    {
      frame->drop_args(args);
      teardown();
      frames.pop_back();
      frame = frames.empty() ? nullptr : &frames.back();
    }
  }

  void Thread::branch(size_t label)
  {
    if (label >= frame->func->labels.size())
      Value::error(Error::BadLabel);

    frame->pc = frame->func->labels.at(label);
  }

  void Thread::check_args(std::vector<uint32_t>& types, bool vararg)
  {
    if ((args < types.size()) || (!vararg && (args > types.size())))
    {
      drop_args();
      Value::error(Error::BadArgs);
    }

    for (size_t i = 0; i < types.size(); i++)
    {
      if (!program->subtype(arg(i)->type_id(), types.at(i)))
      {
        drop_args();
        Value::error(Error::BadType);
      }
    }

    args = 0;
  }

  bool Thread::try_check_args(std::vector<uint32_t>& types)
  {
    if (args != types.size())
    {
      drop_args();
      return false;
    }

    for (size_t i = 0; i < types.size(); i++)
    {
      if (!program->subtype(arg(i)->type_id(), types.at(i)))
      {
        drop_args();
        return false;
      }
    }

    args = 0;
    return true;
  }

  void Thread::check_args(std::vector<Field>& fields)
  {
    if (args != fields.size())
    {
      frame->drop_args(args);
      Value::error(Error::BadArgs);
    }

    for (size_t i = 0; i < args; i++)
    {
      if (!program->subtype(frame->arg(i)->type_id(), fields.at(i).type_id))
      {
        frame->drop_args(args);
        Value::error(Error::BadType);
      }
    }

    args = 0;
  }

  Register& Thread::arg(size_t idx)
  {
    // Only use this when we don't know if we have a frame or not.
    if (frame) [[likely]]
      return frame->arg(idx);

    return locals.at(idx);
  }

  void Thread::drop_args()
  {
    // Only use this when we don't know if we have a frame or not.
    if (frame) [[likely]]
    {
      frame->drop_args(args);
    }
    else
    {
      auto num_args = args;
      args = 0;

      for (size_t i = 0; i < num_args; i++)
        locals.at(i).clear();
    }
  }

  void
  Thread::queue_behavior(Register& result, uint32_t type_id, Function* func)
  {
    if (func->param_types.size() != args)
      Value::error(Error::BadArgs);

    auto& params = func->param_types;
    bool is_closure = false;
    size_t num_cowns = args;
    size_t first_cown = 0;

    if (args > 0)
    {
      auto& closure = frame->arg(0);

      if (!closure->is_cown())
      {
        // The first argument is the closure data.
        is_closure = true;

        if (!program->subtype(closure->type_id(), params.at(0)))
          Value::error(Error::BadArgs);

        if (!closure->is_sendable())
        {
          auto h = closure->get_header();
          auto r = h->region();
          LOG(Error) << "Closure argument is not sendable: " << closure.borrow()
                     << " in region " << r << " stack_rc=" << r->get_stack_rc()
                     << " has_parent=" << r->has_parent()
                     << " has_cown=" << r->has_cown_owner()
                     << " frame_local=" << r->is_frame_local();
          Value::error(Error::BadStackEscape);
        }

        num_cowns--;
        first_cown++;
      }
    }

    // Check that all other args are cowns of the right type.
    for (size_t i = first_cown; i < args; i++)
    {
      auto cown = frame->arg(i)->get_cown();
      auto ref_type = program->ref(cown->content_type_id());

      if (!program->subtype(ref_type, params.at(i)))
        Value::error(Error::BadArgs);
    }

    args = 0;

    if (!program->subtype(func->return_type, program->uncown(type_id)))
      Value::error(Error::BadType);

    // Create the result cown.
    auto result_cown = Cown::create(type_id);

    LOG(Trace) << "Created result cown " << result_cown;

    // Cowns initially have an RC, so we can create a value from it without an
    // incref.
    result = ValueTransfer(result_cown);

    // Request 0 is the result cown.
    auto construction = BehaviourCore::make(
      num_cowns + 1, sizeof(Value) * 2, alignof(Value), run_behavior);
    BehaviourCore::initialise_request(
      construction,
      0,
      result_cown,
      verona::rt::AccessMode::Write,
      verona::rt::Ownership::Borrowed);

    for (size_t i = 0; i < num_cowns; i++)
    {
      // The first cown argument position depends on whether this is a
      // closure or not.
      Value v = frame->arg(first_cown + i).extract();
      auto cown = v.get_cown();
      auto readonly = v.is_readonly();

      // Offset the request by 1 to account for the result cown.
      BehaviourCore::initialise_request(
        construction,
        i + 1,
        cown,
        readonly ? verona::rt::AccessMode::Read : verona::rt::AccessMode::Write,
        verona::rt::Ownership::Transferred);
    }

    auto values = static_cast<Value*>(construction.body);
    new (&values[0]) Value(func);
    new (&values[1]) Value();

    if (is_closure)
    {
      auto& closure = frame->arg(0);

      if (closure->is_header())
      {
        auto r = closure->get_header()->region();

        // Set the region parent, as it's captured by the behavior.
        if (r)
        {
          if (!r->has_owner())
          {
            LOG(Trace) << "Setting closure (" << closure.borrow()
                       << ") region owner: " << r << " to cown owner.";
            r->set_cown_owner(nullptr, closure->get_header());
          }
          else
          {
            LOG(Trace) << "Closure (" << closure.borrow()
                       << ") region already has owner: " << r;
          }
        }
      }

      // Save the closure region before extract clears the register.
      Region* closure_region = nullptr;

      if (closure->is_header())
      {
        auto h = closure->get_header();

        if (h->region() && !h->region()->is_frame_local())
          closure_region = h->region();
      }

      values[1] = closure.extract();

      // The extract moved the value to the behaviour body (raw memory)
      // without reg_dec. Remove the orphaned stack ref — the behaviour
      // dispatch will re-establish it with stack_inc.
      if (closure_region)
      {
        LOG(Trace) << "queue_behavior: stack_dec closure region @"
                   << closure_region;
        closure_region->stack_dec();
      }
    }

    BehaviourCore::schedule(BehaviourCore::finish_construction(construction));
  }

  Register& Thread::get_register(uint64_t idx)
  {
    auto& thread = Thread::get();
    return thread.locals.at(idx);
  }

  void Thread::check_var_type_for_register(uint64_t raw_idx, const Value& v)
  {
    auto& thread = Thread::get();

    // Walk the frame stack to find the frame that owns this register.
    for (auto it = thread.frames.rbegin(); it != thread.frames.rend(); ++it)
    {
      if (raw_idx >= it->base && raw_idx < it->base + it->func->registers)
      {
        auto local_idx = raw_idx - it->base;
        auto params = it->func->param_types.size();
        auto vars = it->func->var_types.size();

        if (local_idx < params || local_idx >= params + vars)
          return;

        auto var_type = it->func->var_types.at(local_idx - params);

        if (var_type == DynamicTypeId)
          return;

        if (!thread.program->subtype(v.type_id(), var_type))
          Value::error(Error::BadType);

        return;
      }
    }
  }

  void
  Thread::thread_handle_callback(CallbackClosure* cc, void* ret, void** args_)
  {
    assert(args == 0);
    auto num_c_args = cc->arg_value_types.size();
    arg(args++) = ValueBorrow(cc->lambda);

    for (size_t i = 0; i < num_c_args; i++)
    {
      auto vt = cc->arg_value_types[i];

      if (vt == ValueType::Dyn)
      {
        auto* val = static_cast<Value*>(args_[i]);
        arg(args++) = ValueBorrow(*val);
      }
      else
      {
        arg(args++) = Value::from_addr(vt, args_[i]);
      }
    }

    try
    {
      auto result = thread_run(cc->func);

      if (cc->return_value_type == ValueType::None)
      {
        return;
      }
      else if (cc->return_value_type == ValueType::Dyn)
      {
        *static_cast<Value*>(ret) = result.extract();
      }
      else
      {
        result.borrow().to_addr(cc->return_value_type, ret);
      }
    }
    catch (Value& error_value)
    {
      {
        logging::Error log;
        log << error_value.to_string() << std::endl;
        print_error_stack(log, error_value);
      }
      std::abort();
    }
  }
}
