#include "program.h"

#include "array.h"
#include "cown.h"
#include "freeze.h"
#include "thread.h"

#include <cstdint>
#include <dlfcn.h>
#include <format>
#include <sched/schedulerthread.h>
#include <zstd.h>

namespace vbci
{
  std::pair<size_t, size_t> SourceFile::linecol(size_t pos)
  {
    // Lines and columns are 0-indexed.
    auto it = std::lower_bound(lines.begin(), lines.end(), pos);

    auto line = it - lines.begin();
    auto col = pos;

    if (it != lines.begin())
      col -= *(it - 1) + 1;

    return {line, col};
  }

  std::string SourceFile::line(size_t line)
  {
    // Lines are 0-indexed.
    if (line > lines.size())
      return {};

    size_t start = 0;
    auto end = contents.size();

    if (line > 0)
      start = lines.at(line - 1) + 1;

    if (line < lines.size())
      end = lines.at(line);

    auto ret = contents.substr(start, end - start);
    std::erase(ret, '\r');
    return ret;
  }

  Program& Program::get()
  {
    static Program program;
    return program;
  }

  Symbol& Program::symbol(size_t idx)
  {
    if (idx >= symbols.size())
      Value::error(Error::UnknownFFI);

    return symbols.at(idx);
  }

  Function* Program::function(size_t idx)
  {
    if (idx >= functions.size())
      return nullptr;

    return &functions.at(idx);
  }

  Class& Program::cls(uint32_t type_id)
  {
    if (type_id >= classes.size())
      Value::error(Error::BadType);

    return classes.at(type_id);
  }

  ComplexType& Program::complex_type(uint32_t type_id)
  {
    if (!is_complex(type_id))
      Value::error(Error::BadType);

    return complex_types.at(type_id - min_complex_type_id);
  }

  ffi_type* Program::value_type()
  {
    return &ffi_type_value;
  }

  Register& Program::memo_slot(size_t index)
  {
    return memo_slots.at(index);
  }

  void Program::init_memo_slot(size_t index)
  {
    auto& slot = memo_slots.at(index);
    assert(slot->is_invalid());
    slot = Thread::run_sync(&functions.at(memo_func_ids.at(index)));

    if (slot->is_header())
      freeze(slot->get_header());

    slot.borrow().immortalize();
  }

  uint32_t Program::get_typeid_arg()
  {
    return typeid_arg;
  }

  uint32_t Program::get_typeid_argv()
  {
    return typeid_argv;
  }

  uint32_t Program::get_typeid_array_usize()
  {
    return typeid_array_usize;
  }

  uint32_t Program::get_typeid_ffi_struct_result()
  {
    return typeid_ffi_struct_result;
  }

  Array* Program::get_argv()
  {
    return argv;
  }

  Array* Program::get_string(size_t idx)
  {
    if (idx >= string_arrays.size())
      Value::error(Error::UnknownString);

    return string_arrays.at(idx);
  }

  int Program::run(
    std::filesystem::path& path,
    size_t num_threads,
    std::vector<std::string> args)
  {
    file = path;

    if (!load())
      return -1;

    setup_strings();
    setup_argv(args);

    auto& sched = verona::rt::Scheduler::get();
    sched.init(num_threads);

    // Pre-size memo slots so initializers can load earlier slots by index.
    memo_slots.resize(memo_func_ids.size());

    // Run memo initializers in declaration order. Any memo loaded by an
    // initializer must therefore have been declared earlier.
    // This must happen after sched.init() because once functions may create
    // cowns (via `when`), which requires the scheduler's core pool to be
    // initialized for behavior queuing.
    for (size_t i = 0; i < memo_func_ids.size(); i++)
      init_memo_slot(i);

    ValueTransfer ret =
      Thread::run_async(typeid_cown_none, &functions.at(MainFunctionId));
    sched.run();

    auto ret_val = ret.get_cown()->load();
    ret.field_dec();

    if (ret_val.is_error())
    {
      LOG(Error) << ret_val.to_string();

      if (exit_code == 0)
        exit_code = -1;
    }

    // Once values and their reachable object graphs have process lifetime.
    // Clearing the slots is safe because their values have been immortalized.
    memo_slots.clear();

    cleanup_strings();

    return exit_code;
  }

  std::pair<ValueType, ffi_type*> Program::layout_type_id(uint32_t type_id)
  {
    if (type_id == DynamicTypeId)
    {
      // Dynamic.
      return {ValueType::Dyn, &ffi_type_value};
    }
    else if (type_id < NumPrimitiveClasses)
    {
      // Primitive type.
      switch (PrimitiveType(type_id))
      {
        case PrimitiveType::None:
          return {ValueType::None, &ffi_type_void};
        case PrimitiveType::Bool:
          return {ValueType::Bool, &ffi_type_uint8};
        case PrimitiveType::I8:
          return {ValueType::I8, &ffi_type_sint8};
        case PrimitiveType::I16:
          return {ValueType::I16, &ffi_type_sint16};
        case PrimitiveType::I32:
          return {ValueType::I32, &ffi_type_sint32};
        case PrimitiveType::I64:
          return {ValueType::I64, &ffi_type_sint64};
        case PrimitiveType::U8:
          return {ValueType::U8, &ffi_type_uint8};
        case PrimitiveType::U16:
          return {ValueType::U16, &ffi_type_uint16};
        case PrimitiveType::U32:
          return {ValueType::U32, &ffi_type_uint32};
        case PrimitiveType::U64:
          return {ValueType::U64, &ffi_type_uint64};
        case PrimitiveType::F32:
          return {ValueType::F32, &ffi_type_float};
        case PrimitiveType::F64:
          return {ValueType::F64, &ffi_type_double};
        case PrimitiveType::ILong:
          return {ValueType::ILong, &ffi_type_slong};
        case PrimitiveType::ULong:
          return {ValueType::ULong, &ffi_type_ulong};
        case PrimitiveType::ISize:
          return {
            ValueType::ISize,
            sizeof(ssize_t) == 4 ? &ffi_type_sint32 : &ffi_type_sint64};
        case PrimitiveType::USize:
          return {
            ValueType::USize,
            sizeof(size_t) == 4 ? &ffi_type_uint32 : &ffi_type_uint64};
        case PrimitiveType::Ptr:
          return {ValueType::Ptr, &ffi_type_pointer};
        default:
          break;
      }
    }
    else if (type_id < min_complex_type_id)
    {
      // Class type.
      return {ValueType::Object, &ffi_type_pointer};
    }
    else if (type_id < (min_complex_type_id + complex_types.size()))
    {
      // Complex type.
      auto& c = complex_types.at(type_id - min_complex_type_id);

      switch (c.tag)
      {
        case TypeTag::Array:
          return {ValueType::Array, &ffi_type_pointer};
        case TypeTag::Cown:
          return {ValueType::Cown, &ffi_type_pointer};
        case TypeTag::Ref:
          return {ValueType::Dyn, &ffi_type_value};
        case TypeTag::Union:
          return layout_union_type(c);
        case TypeTag::Tuple:
          return {ValueType::Array, &ffi_type_pointer};
        default:
          break;
      }
    }

    assert(false);
    return {ValueType::Invalid, &ffi_type_void};
  }

  std::pair<ValueType, ffi_type*> Program::layout_union_type(ComplexType& t)
  {
    assert(t.tag == TypeTag::Union);

    // Bottom type: no value can inhabit an empty union. Raise a runtime
    // error rather than producing an invalid layout. An empty union
    // reaching this path indicates a compiler bug (typically reify
    // emitting a shape-class with no concrete implementors that is still
    // referenced by a reachable function's signature).
    if (t.children.empty())
      Value::error(Error::BadType);

    auto& child = t.children.at(0);
    auto rep = layout_type_id(child);
    bool ok = true;

    for (size_t i = 1; i < t.children.size(); i++)
    {
      if (rep != layout_type_id(t.children.at(i)))
      {
        ok = false;
        break;
      }
    }

    // If all elements of the union have the same representation, use it.
    if (ok)
      return rep;

    // Otherwise, use the generic Value representation.
    return {ValueType::Dyn, &ffi_type_value};
  }

  bool Program::is_complex(uint32_t type_id)
  {
    return type_id >= min_complex_type_id;
  }

  bool Program::is_array(uint32_t type_id)
  {
    return is_complex(type_id) &&
      (complex_type(type_id).tag == TypeTag::Array ||
       complex_type(type_id).tag == TypeTag::Tuple);
  }

  // Used by runtime Typetest instructions for tuple type checking.
  bool Program::is_tuple(uint32_t type_id)
  {
    return is_complex(type_id) && (complex_type(type_id).tag == TypeTag::Tuple);
  }

  bool Program::is_ref(uint32_t type_id)
  {
    return is_complex(type_id) && (complex_type(type_id).tag == TypeTag::Ref);
  }

  bool Program::is_cown(uint32_t type_id)
  {
    return is_complex(type_id) && (complex_type(type_id).tag == TypeTag::Cown);
  }

  bool Program::is_union(uint32_t type_id)
  {
    return is_complex(type_id) && (complex_type(type_id).tag == TypeTag::Union);
  }

  uint32_t Program::unarray(uint32_t type_id)
  {
    auto& t = complex_type(type_id);

    if (t.tag == TypeTag::Tuple)
      return DynamicTypeId;

    if (t.tag != TypeTag::Array)
      Value::error(Error::BadType);

    return t.children.at(0);
  }

  uint32_t Program::uncown(uint32_t type_id)
  {
    auto& t = complex_type(type_id);

    if (t.tag != TypeTag::Cown)
      Value::error(Error::BadType);

    return t.children.at(0);
  }

  uint32_t Program::unref(uint32_t type_id)
  {
    auto& t = complex_type(type_id);

    if (t.tag != TypeTag::Ref)
      Value::error(Error::BadType);

    return t.children.at(0);
  }

  uint32_t Program::ref(uint32_t type_id)
  {
    auto it = ref_map.find(type_id);

    if (it == ref_map.end())
      return typeid_ref_dyn;

    return it->second;
  }

  bool Program::subtype(uint32_t sub, uint32_t super)
  {
    // Everything is a subtype of dynamic.
    if (super == DynamicTypeId)
      return true;

    // Dynamic is a subtype of nothing.
    if (sub == DynamicTypeId)
      return false;

    // If it's the same, we're done.
    if (sub == super)
      return true;

    if (is_union(sub))
    {
      // All elements of sub must be subtypes of super.
      auto& sub_t = complex_type(sub);

      for (auto c : sub_t.children)
      {
        if (!subtype(c, super))
          return false;
      }

      return true;
    }

    if (is_union(super))
    {
      // Sub must be a subtype of one of the elements of super.
      auto& super_t = complex_type(super);

      for (auto c : super_t.children)
      {
        if (subtype(sub, c))
          return true;
      }

      return false;
    }

    if (is_complex(sub))
    {
      // Sub is an array, cown, ref, or tuple.
      if (!is_complex(super))
        return false;

      auto& sub_t = complex_type(sub);
      auto& super_t = complex_type(super);

      // Tuple vs Tuple: covariant element-wise, same arity.
      // Covariance is safe because tuples are read-only after construction
      // (constructed via NewArrayConst + Store, read via ArrayRefConst + Load).
      if (sub_t.tag == TypeTag::Tuple && super_t.tag == TypeTag::Tuple)
      {
        if (sub_t.children.size() != super_t.children.size())
          return false;

        for (size_t i = 0; i < sub_t.children.size(); i++)
        {
          if (!subtype(sub_t.children[i], super_t.children[i]))
            return false;
        }

        return true;
      }

      // Tuple vs Array: tuple is a subtype of Array(T) if all element
      // types are subtypes of T.
      if (sub_t.tag == TypeTag::Tuple && super_t.tag == TypeTag::Array)
      {
        auto super_elem = super_t.children.at(0);

        for (auto c : sub_t.children)
        {
          if (!subtype(c, super_elem))
            return false;
        }

        return true;
      }

      if (sub_t.tag != super_t.tag)
        return false;

      auto sub_elem = sub_t.children.at(0);
      auto super_elem = super_t.children.at(0);
      return subtype(sub_elem, super_elem) && subtype(super_elem, sub_elem);
    }

    return false;
  }

  std::string Program::debug_info(Function* func, PC pc)
  {
    if (!di_decompress())
      return std::format(" --> {}:{}", fallback_function(func), pc);

    constexpr auto no_value = size_t(-1);
    auto di_file = no_value;
    auto di_offset = 0;
    auto cur_pc = func->labels.at(0);
    auto di_pc = di + func->debug_info;

    // Read past the function name and the register names.
    di_uleb(di_pc);

    for (size_t i = 0; i < func->registers; i++)
      di_uleb(di_pc);

    while (cur_pc <= pc)
    {
      auto u = di_uleb(di_pc);
      auto op = u & 0x03;
      u >>= 2;

      if (op == +DIOp::File)
      {
        di_file = u;
        di_offset = 0;
      }
      else if (op == +DIOp::Offset)
      {
        di_offset += u;
        cur_pc++;
      }
      else if (op == +DIOp::Skip)
      {
        cur_pc += u;
      }
      else
      {
        break;
      }
    }

    if (di_file == no_value)
      return std::format(" --> {}:{}", fallback_function(func), di_offset);

    auto& filename = di_strings.at(di_file);
    auto source = get_source_file(di_file);

    if (!source)
      return std::format(" --> {}:{}", filename, di_offset);

    auto [line, col] = source->linecol(di_offset);
    auto src_line = source->line(line);
    auto caret = src_line.substr(0, col);

    std::replace_if(
      caret.begin(),
      caret.end(),
      [](unsigned char ch) { return ch != '\t'; },
      ' ');

    return std::format(
      " --> {}:{}:{}\n  | {}\n  | {}^",
      filename,
      line + 1,
      col + 1,
      src_line,
      caret);
  }

  std::string Program::di_function(Function* func)
  {
    if (!di_decompress())
      return fallback_function(func);

    auto pc = di + func->debug_info;
    return di_strings.at(di_uleb(pc));
  }

  std::string Program::fallback_function(Function* func)
  {
    if (func == nullptr)
      return "function <null>";

    if (functions.empty())
      return "function <unloaded>";

    auto first = reinterpret_cast<uintptr_t>(functions.data());
    auto ptr = reinterpret_cast<uintptr_t>(func);
    auto size = functions.size() * sizeof(Function);

    if ((ptr >= first) && (ptr < (first + size)))
    {
      auto offset = ptr - first;

      if ((offset % sizeof(Function)) == 0)
        return std::format("function#{}", offset / sizeof(Function));
    }

    return "function <external>";
  }

  std::string Program::di_class(Class& cls)
  {
    if (!di_decompress())
      return std::format("class {}", cls.type_id);

    auto pc = di + cls.debug_info;
    return di_strings.at(di_uleb(pc));
  }

  std::string Program::di_field(Class& cls, size_t idx)
  {
    if (!di_decompress())
      return std::to_string(idx);

    auto pc = di + cls.debug_info;
    di_uleb(pc);

    while (idx > 0)
      di_uleb(pc);

    return di_strings.at(di_uleb(pc));
  }

  void Program::setup_strings()
  {
    string_arrays.resize(strings.size());

    for (size_t i = 0; i < strings.size(); i++)
    {
      auto& str = strings.at(i);
      auto str_size = str.size() + 1;
      auto arr = Array::create(
        new uint8_t[Array::size_of(str_size, ffi_type_uint8.size)],
        Location::stack(),
        typeid_arg,
        ValueType::U8,
        str_size,
        ffi_type_uint8.size);

      auto p = arr->get_pointer();
      std::memcpy(p, str.c_str(), str_size);
      // Array size honestly includes the null terminator.
      // The string wrapper (in Verona) sets len = data.size - 1.
      arr->immortalize();
      string_arrays[i] = arr;
    }
  }

  void Program::cleanup_strings()
  {
    for (auto* arr : string_arrays)
      delete[] reinterpret_cast<uint8_t*>(arr);

    string_arrays.clear();
  }

  void Program::setup_value_type()
  {
    ffi_type_value_elements.clear();
    ffi_type_value_elements.push_back(&ffi_type_uint64);
    ffi_type_value_elements.push_back(&ffi_type_uint64);
    ffi_type_value_elements.push_back(nullptr);

    ffi_type_value.size = sizeof(Value);
    ffi_type_value.alignment = alignof(Value);
    ffi_type_value.type = FFI_TYPE_STRUCT;
    ffi_type_value.elements = ffi_type_value_elements.data();
  }

  void Program::setup_argv(std::vector<std::string>& args)
  {
    auto arg_rep = layout_type_id(unarray(typeid_arg));
    auto argv_rep = layout_type_id(unarray(typeid_argv));

    argv = Array::create(
      new uint8_t[Array::size_of(args.size(), argv_rep.second->size)],
      Location::stack(),
      typeid_argv,
      argv_rep.first,
      args.size(),
      argv_rep.second->size);

    for (size_t i = 0; i < args.size(); i++)
    {
      auto str = args.at(i);
      auto str_size = str.size() + 1;
      auto arg = Array::create(
        new uint8_t[Array::size_of(str_size, arg_rep.second->size)],
        Location::stack(),
        typeid_arg,
        arg_rep.first,
        str_size,
        arg_rep.second->size);

      auto p = arg->get_pointer();
      std::memcpy(p, str.c_str(), str_size);
      arg->set_size(str_size - 1);

      // TODO Register use here is pointless, and we should optimise.
      // Create a exchange that doesn't use reference counting.
      Register val = ValueTransfer(arg);
      Register dst = argv->exchange<true>(i, std::move(val));
    }

    argv->immortalize();
  }

  bool Program::load()
  {
    content.clear();
    functions.clear();
    classes.clear();
    string_arrays.clear();

    libs.clear();
    symbols.clear();
    setup_value_type();

    argv = nullptr;

    di = PC(-1);
    di_strings.clear();
    source_files.clear();

    std::ifstream f(file, std::ios::binary | std::ios::in | std::ios::ate);

    if (!f)
    {
      LOG(Error) << file << ": couldn't load" << std::endl;
      return false;
    }

    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    content.resize(size);
    f.read(reinterpret_cast<char*>(&content.at(0)), size);

    if (!f)
    {
      LOG(Error) << file << ": couldn't read" << std::endl;
      return false;
    }

    PC pc = 0;

    if (uleb(pc) != MagicNumber)
    {
      LOG(Error) << file << ": does not start with the magic number"
                 << std::endl;
      return false;
    }

    if (uleb(pc) != CurrentVersion)
    {
      LOG(Error) << file << ": has an unknown version number" << std::endl;
      return false;
    }

    // String table.
    string_table(pc, content, strings);
    auto num_classes = uleb(pc);
    auto num_complex_primitives = uleb(pc);
    classes.resize(NumPrimitiveClasses + num_classes + num_complex_primitives);
    min_complex_type_id = NumPrimitiveClasses + num_classes;
    uint32_t class_id = 0;

    // Primitive classes.
    for (size_t i = 0; i < NumPrimitiveClasses; i++)
    {
      auto& cls = classes.at(class_id);
      cls.type_id = class_id++;
      cls.debug_info = size_t(-1);

      if (!parse_methods(cls, pc))
        return false;
    }

    // User-defined classes.
    for (size_t i = 0; i < num_classes; i++)
    {
      auto& cls = classes.at(class_id);
      cls.type_id = class_id++;
      cls.debug_info = uleb(pc);

      if (!parse_fields(cls, pc))
        return false;

      if (!parse_methods(cls, pc))
        return false;
    }

    // Complex primitive classes.
    for (size_t i = 0; i < num_complex_primitives; i++)
    {
      auto& cls = classes.at(class_id);
      cls.type_id = class_id++;
      cls.debug_info = size_t(-1);

      if (!parse_methods(cls, pc))
        return false;
    }

    // FFI information.
    auto num_libs = uleb(pc);
    libs.reserve(num_libs);
    for (size_t i = 0; i < num_libs; i++)
      libs.emplace_back(strings.at(uleb(pc)));

    auto num_symbols = uleb(pc);
    for (size_t i = 0; i < num_symbols; i++)
    {
      auto& lib = libs.at(uleb(pc));
      auto& name = strings.at(uleb(pc));
      auto& version = strings.at(uleb(pc));
      bool vararg = uleb(pc);
      auto func = lib.symbol(name, version);

      if (!func)
      {
        LOG(Error) << file << ": couldn't load symbol " << name << "@"
                   << version << std::endl;
        return false;
      }

      symbols.push_back(Symbol(func, vararg));
      auto& symbol = symbols.back();

      auto num_params = uleb(pc);
      for (size_t j = 0; j < num_params; j++)
        symbol.param(uleb(pc));

      symbol.ret(uleb(pc));
    }

    // Functions.
    functions.resize(uleb(pc));

    if (functions.empty())
    {
      LOG(Error) << file << ": has no functions" << std::endl;
      return false;
    }

    for (auto& func : functions)
    {
      if (!parse_function(func, pc))
        return false;
    }

    if (functions.at(MainFunctionId).param_types.size() != 0)
    {
      LOG(Error) << file << ": `main` must take zero parameters" << std::endl;
      return false;
    }

    if (!subtype(functions.at(MainFunctionId).return_type, +ValueType::None))
    {
      LOG(Error) << file << ": `main` must return none" << std::endl;
      return false;
    }

    // Complex types.
    complex_types.resize(uleb(pc));
    uint32_t type_id = min_complex_type_id;

    for (auto& t : complex_types)
      parse_complex_type(t, type_id++, pc);

    typeid_cown_none = min_complex_type_id;
    assert(complex_type(typeid_cown_none).tag == TypeTag::Cown);
    assert(complex_type(typeid_cown_none).children.at(0) == +ValueType::None);

    typeid_arg = min_complex_type_id + 1;
    assert(complex_type(typeid_arg).tag == TypeTag::Array);
    assert(complex_type(typeid_arg).children.at(0) == +ValueType::U8);

    typeid_argv = min_complex_type_id + 2;
    assert(complex_type(typeid_argv).tag == TypeTag::Array);
    assert(complex_type(typeid_argv).children.at(0) == typeid_arg);

    typeid_ref_dyn = min_complex_type_id + 3;
    assert(complex_type(typeid_ref_dyn).tag == TypeTag::Ref);
    assert(complex_type(typeid_ref_dyn).children.at(0) == DynamicTypeId);

    typeid_array_usize = min_complex_type_id + 4;
    assert(complex_type(typeid_array_usize).tag == TypeTag::Array);
    assert(
      complex_type(typeid_array_usize).children.at(0) == +ValueType::USize);

    typeid_ffi_struct_result = min_complex_type_id + 5;
    assert(complex_type(typeid_ffi_struct_result).tag == TypeTag::Tuple);
    assert(complex_type(typeid_ffi_struct_result).children.size() == 3);
    assert(
      complex_type(typeid_ffi_struct_result).children.at(0) ==
      +ValueType::USize);
    assert(
      complex_type(typeid_ffi_struct_result).children.at(1) ==
      typeid_array_usize);
    assert(complex_type(typeid_ffi_struct_result).children.at(2) == typeid_arg);

    // Prepare symbols now that all type information is available.
    for (auto& symbol : symbols)
    {
      if (!symbol.prepare())
      {
        LOG(Error) << file << ": couldn't prepare symbol" << std::endl;
        return false;
      }
    }

    for (auto& cls : classes)
    {
      if (!fixup_methods(cls))
        return false;
    }

    // Function label locations are relative to the code section. Make them
    // absolute.
    auto memo_count = uleb(pc);
    memo_func_ids.resize(memo_count);
    for (size_t i = 0; i < memo_count; i++)
      memo_func_ids[i] = uleb(pc);

    auto code_size = uleb(pc);

    for (auto& func : functions)
    {
      for (auto& label : func.labels)
        label += pc;
    }

    // Debug info.
    pc += code_size;

    if (pc < content.size())
      di = pc;
    else
      di = PC(-1);

    return true;
  }

  bool Program::parse_function(Function& f, PC& pc)
  {
    f.registers = uleb(pc);
    f.debug_info = uleb(pc);

    // Function signature.
    auto params = uleb(pc);
    f.param_types.resize(params);

    for (size_t i = 0; i < params; i++)
      f.param_types.at(i) = uleb(pc);

    f.return_type = uleb(pc);

    // Variable types.
    auto vars = uleb(pc);
    f.var_types.resize(vars);

    for (size_t i = 0; i < vars; i++)
      f.var_types.at(i) = uleb(pc);

    // Labels.
    f.labels.resize(uleb(pc));

    if (f.labels.empty())
    {
      LOG(Error) << file << ": function has no labels" << std::endl;
      return false;
    }

    for (auto& label : f.labels)
      label = uleb(pc);

    return true;
  }

  bool Program::parse_fields(Class& cls, PC& pc)
  {
    auto num_fields = uleb(pc);
    cls.field_map.reserve(num_fields);
    cls.fields.resize(num_fields);

    for (size_t i = 0; i < num_fields; i++)
    {
      auto& f = cls.fields.at(i);
      cls.field_map.emplace(uleb(pc), i);
      f.type_id = uleb(pc);
    }

    return true;
  }

  bool Program::parse_methods(Class& cls, PC& pc)
  {
    auto num_methods = uleb(pc);
    cls.methods.reserve(num_methods);

    for (size_t i = 0; i < num_methods; i++)
    {
      auto method_id = uleb(pc);
      auto func_id = uleb(pc);
      cls.methods.emplace(method_id, reinterpret_cast<Function*>(func_id));
    }

    return true;
  }

  bool Program::fixup_methods(Class& cls)
  {
    for (auto& method : cls.methods)
    {
      auto idx = reinterpret_cast<size_t>(method.second);
      auto& func = functions.at(idx);
      method.second = &func;

      if (method.first == FinalizerMethodId)
      {
        if (func.param_types.size() != 1)
        {
          LOG(Error) << file << ": finalizer must have one parameter"
                     << std::endl;
          return false;
        }
      }
    }

    // Calculate the class size.
    try
    {
      if (!cls.calc_size())
      {
        LOG(Error) << file << ": couldn't calculate class size" << std::endl;
        return false;
      }
    }
    catch (const Value& err)
    {
      LOG(Error) << file
                 << ": couldn't calculate class size: " << err.to_string()
                 << std::endl;
      return false;
    }

    return true;
  }

  void Program::parse_complex_type(ComplexType& t, uint32_t type_id, PC& pc)
  {
    t.tag = TypeTag(uleb(pc));

    switch (t.tag)
    {
      case TypeTag::Array:
      case TypeTag::Cown:
      {
        t.children.push_back(uleb(pc));
        break;
      }

      case TypeTag::Ref:
      {
        auto child = uleb(pc);
        t.children.push_back(child);
        ref_map[child] = type_id;
        break;
      }

      case TypeTag::Union:
      {
        auto size = uleb(pc);

        for (size_t i = 0; i < size; i++)
          t.children.push_back(uleb(pc));
        break;
      }

      case TypeTag::Tuple:
      {
        auto size = uleb(pc);

        for (size_t i = 0; i < size; i++)
          t.children.push_back(uleb(pc));
        break;
      }
    }
  }

  std::string Program::str(size_t& pc, std::vector<uint8_t>& from)
  {
    auto size = uleb(pc, from);
    auto str = std::string(from.begin() + pc, from.begin() + pc + size);
    pc += size;
    return str;
  }

  void Program::string_table(
    size_t& pc, std::vector<uint8_t>& from, std::vector<std::string>& table)
  {
    auto count = uleb(pc, from);
    table.clear();
    table.reserve(count);

    for (size_t i = 0; i < count; i++)
      table.push_back(str(pc, from));
  }

  bool Program::di_decompress()
  {
    std::call_once(di_once, [this]() {
      if (di == PC(-1))
        return;

      auto cap = ZSTD_getFrameContentSize(&content.at(di), content.size() - di);

      if ((cap == ZSTD_CONTENTSIZE_ERROR) || (cap == ZSTD_CONTENTSIZE_UNKNOWN))
        return;

      di_content.resize(cap);
      auto decompressed_size = ZSTD_decompress(
        di_content.data(), cap, &content.at(di), content.size() - di);

      if (ZSTD_isError(decompressed_size))
      {
        di_content.clear();
        return;
      }

      di_content.resize(decompressed_size);

      PC pc = 0;
      string_table(pc, di_content, di_strings);
      auto num_sources = di_uleb(pc);

      for (size_t i = 0; i < num_sources; i++)
      {
        auto di_file = di_uleb(pc);
        source_files[di_file].di_pos = pc;
        pc += di_uleb(pc);
      }

      di = pc;
    });

    return !di_content.empty();
  }

  SourceFile* Program::get_source_file(size_t di_file)
  {
    auto find = source_files.find(di_file);
    if (find == source_files.end())
      return nullptr;

    auto& source = find->second;

    if (source.contents.empty())
    {
      source.contents = str(source.di_pos, di_content);
      auto pos = source.contents.find('\n');

      while (pos != std::string::npos)
      {
        source.lines.push_back(pos);
        pos = source.contents.find('\n', pos + 1);
      }
    }

    return &source;
  }
}
