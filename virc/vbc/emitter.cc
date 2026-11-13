#include "emitter.h"

#include "../lang.h"
#include "debug_info.h"
#include "encoder.h"
#include "instruction_encoder.h"
#include "string_table.h"
#include "type_encoding.h"

#include <vbc/format.h>

namespace virc
{
  using namespace ::vbc;
  using namespace vbc_backend;
  using ::vbc::CurrentVersion;
  using ::vbc::MagicNumber;

  namespace
  {
    struct VBCEmitter : Compilation
    {
      explicit VBCEmitter(const Compilation& compilation)
      : Compilation(compilation)
      {}

      void emit(std::filesystem::path output, bool strip);
    };
  }

  void VBCEmitter::emit(std::filesystem::path output, bool strip)
  {
    wf::push_back(wfIR);

    if (output.empty())
      output = "out.vbc";

    ByteBuffer header;
    ByteBuffer code;
    DebugInfo debug{code, source_paths};

    MemoSlots memo_slots;
    std::vector<Node> memo_initializers;

    for (auto& child : *top)
    {
      if (child != Memo)
        continue;

      auto slot = std::string((child / MemoId)->location().view());
      memo_slots[slot] = memo_initializers.size();
      memo_initializers.push_back(child / FunctionId);
    }

    header << uleb(MagicNumber);
    header << uleb(CurrentVersion);
    encode_string_table(header, ST::exec());

    header << uleb(classes.size());
    header << uleb(complex_primitives.size());

    for (auto& primitive : primitives)
    {
      if (primitive)
      {
        auto methods = primitive / Methods;
        header << uleb(methods->size());

        for (auto& method : *methods)
        {
          header << uleb(*get_method_id(method / MethodId))
                 << uleb(*get_func_id(method / FunctionId));
        }
      }
      else
      {
        header << uleb(0);
      }
    }

    for (auto& class_node : classes)
    {
      header << uleb(debug.size());
      debug.output() << uleb(ST::di().string(class_node / ClassId));

      auto fields = class_node / Fields;
      header << uleb(fields->size());

      for (auto& field : *fields)
      {
        header << uleb(*get_field_id(field / FieldId));
        header << uleb(type_id(field / Type));
        debug.output() << uleb(ST::di().string(field / FieldId));
      }

      auto methods = class_node / Methods;
      header << uleb(methods->size());

      for (auto& method : *methods)
      {
        header << uleb(*get_method_id(method / MethodId))
               << uleb(*get_func_id(method / FunctionId));
        debug.output() << uleb(ST::di().string(method / MethodId));
      }
    }

    for (auto& primitive : complex_primitives)
    {
      if (primitive)
      {
        auto methods = primitive / Methods;
        header << uleb(methods->size());

        for (auto& method : *methods)
        {
          header << uleb(*get_method_id(method / MethodId))
                 << uleb(*get_func_id(method / FunctionId));
        }
      }
      else
      {
        header << uleb(0);
      }
    }

    header << uleb(libraries.size());

    for (auto& library : libraries)
      header << uleb(ST::exec().string(library / String));

    header << uleb(symbols.size());

    for (auto& symbol : symbols)
    {
      header << uleb(*get_library_id(symbol->parent(Lib)))
             << uleb(ST::exec().string(symbol / Lhs))
             << uleb(ST::exec().string(symbol / Rhs))
             << uleb(((symbol / Vararg) == Vararg) ? 1 : 0)
             << uleb((symbol / FFIParams)->size());

      for (auto& parameter : *(symbol / FFIParams))
        header << uleb(type_id(parameter));

      header << uleb(type_id(symbol / Return));
    }

    header << uleb(functions.size());

    for (auto& function : functions)
    {
      header << uleb(function.register_idxs.size());
      header << uleb(debug.size());
      header << uleb(function.params);

      for (auto& parameter : *(function.func / Params))
        header << uleb(type_id(parameter / Type));

      header << uleb(type_id(function.func / Type));

      auto variables = function.func / Vars;
      header << uleb(variables->size());

      for (auto& variable : *variables)
        header << uleb(type_id(variable / Type));

      header << uleb(function.label_idxs.size());
      debug.output() << uleb(function.name);

      for (auto& name : function.register_names)
        debug.output() << uleb(name);

      debug.begin_function();

      for (auto label : *(function.func / Labels))
      {
        header << uleb(code.size());

        for (Node statement : *(label / Body))
        {
          if (statement == vir::Source)
          {
            debug.record_explicit_file(statement / String);
            continue;
          }

          if (statement == Offset)
          {
            debug.record_explicit_offset(
              from_chars_sep_v<size_t>(statement / Int));
            continue;
          }

          debug.record_statement(statement);
          encode_statement(*this, function, memo_slots, code, statement);
        }

        Node terminator = label / Return;
        debug.finish_function(terminator);
        encode_terminator(*this, function, code, terminator);
      }
    }

    encode_type_table(header, types);

    header << uleb(memo_initializers.size());
    for (auto& function_id : memo_initializers)
      header << uleb(*get_func_id(function_id));

    header << uleb(code.size());
    std::ofstream file(output, std::ios::binary | std::ios::out);
    file.write(
      reinterpret_cast<const char*>(header.data()),
      static_cast<std::streamsize>(header.size()));
    file.write(
      reinterpret_cast<const char*>(code.data()),
      static_cast<std::streamsize>(code.size()));

    if (!strip)
      debug.write_to(file, output);

    if (!file)
      logging::Error() << "Error writing to: " << output << std::endl;

    wf::pop_front();
  }

  void vbc_backend::emit(
    const Compilation& compilation,
    const std::filesystem::path& output,
    bool strip)
  {
    VBCEmitter(compilation).emit(output, strip);
  }

}

void virc::vbc::emit(
  const virc::Compilation& compilation,
  const std::filesystem::path& output,
  bool strip)
{
  virc::vbc_backend::emit(compilation, output, strip);
}
