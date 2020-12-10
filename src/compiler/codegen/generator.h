// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/helpers.h"
#include "interpreter/bytecode.h"

#include <functional>
#include <iostream>
#include <optional>
#include <vector>

namespace verona::compiler
{
  struct Label;
  struct Descriptor;
  struct CalleeRegister;

  namespace detail
  {
    template<typename T>
    struct is_enum_class
    : public std::
        conjunction<std::is_enum<T>, std::negation<std::is_convertible<T, int>>>
    {};
    template<typename T>
    static constexpr bool is_enum_class_v = is_enum_class<T>::value;
  }

  /**
   * Bytecode generation primitives.
   *
   * The generator allows writing relocatable values, whose value are not yet
   * known. For each such value, a Relocatable handle is obtained using
   * `create_relocatable`.
   *
   * The handle can be written using Generator's overloaded methods that take
   * Relocatable arguments. Space will be made in the output data, and a
   * relocation will be registered.
   *
   * `define_relocatable` is used once the actual value is known. However
   * relocations are not actually resolved until `finish` is called.
   *
   */
  class Generator
  {
  public:
    struct Relocatable;
    typedef uint64_t RelocationValue;

    Generator(std::vector<uint8_t>& code) : code_(code) {}

    /**
     * Write integer values in little endian format.
     */
    void u8(uint8_t value);
    void u16(uint16_t value);
    void u32(uint32_t value);
    void u64(uint64_t value);

    /**
     * Write a string as a 2-byte length followed by the string data.
     */
    void str(std::string_view s);

    void opcode(bytecode::Opcode opcode);
    void selector(bytecode::SelectorIdx index);

    /**
     * Write relocatable integer values in little endian format, with an
     * optional added.
     *
     * After linking, the value will be the label's value relative to
     * "relative_to"
     */
    void u8(Relocatable relocatable, size_t relative_to = 0);
    void u16(Relocatable relocatable, size_t relative_to = 0);
    void s16(Relocatable relocatable, size_t relative_to = 0);
    void u32(Relocatable relocatable, size_t relative_to = 0);
    void u64(Relocatable relocatable, size_t relative_to = 0);

    /**
     * Create a new relocatable handle.
     */
    Relocatable create_relocatable();
    Label create_label();
    Descriptor create_descriptor();

    /**
     * Set the value for a relocatable handle.
     *
     * This should only be called once per handle. The relocations that refer to
     * this handle will not fixed until `finish` is called as well.
     */
    void define_relocatable(Relocatable relocatable, RelocationValue value);

    /**
     * Define a label to the current offset.
     */
    void define_label(Label label);

    /**
     * Link the bytecode by resolving all relocations.
     *
     * This must only be called once, after all bytecode has been written and
     * all labels are defined.
     */
    void finish();

    size_t current_offset()
    {
      return code_.size();
    }

    /**
     * Emit an instruction to the bytecode.
     *
     * The arguments passed to this function should match the type of the
     * opcode's operands, as defined in the opcode spec in
     * `interpreter/bytecode.h`.
     */
    template<bytecode::Opcode Op, typename... Args>
    void emit(Args&&... args)
    {
      using Operands = typename bytecode::OpcodeSpec<Op>::Operands;
      emit_impl<Op>(Operands{}, std::forward<Args>(args)...);
    }

  private:
    template<
      bytecode::Opcode Op,
      typename... Operands,
      typename... Args,
      typename = std::enable_if_t<sizeof...(Operands) == sizeof...(Args)>>
    void emit_impl(bytecode::OpcodeOperands<Operands...>, Args&&... args)
    {
      size_t opcode_start = current_offset();
      opcode(Op);
      std::initializer_list<int> x{
        (emit_helper<Operands>::write(
           this, opcode_start, std::forward<Args>(args)),
         0)...};
    }

    /**
     * Specialization hook, used to emit values based on the expected type
     * listed in the opcode specification.
     */
    template<typename T, typename = void>
    struct emit_helper;

    /**
     * Write an integer value in little endian format.
     *
     * The common_type_t disables template parameter deduction, forcing the
     * caller the specify the integer type explicitly.
     */
    template<typename T>
    std::enable_if_t<std::is_integral_v<T>> write(std::common_type_t<T> value)
    {
      size_t offset = code_.size();
      code_.reserve(offset + sizeof(T));
      for (size_t i = 0; i < sizeof(T) * 8; i += 8)
      {
        code_.push_back((value >> i) & 0xff);
      }
    }

    void add_relocation(
      size_t offset,
      uint8_t width,
      Relocatable relocatable,
      size_t relative_to,
      bool is_signed);

    struct Relocation
    {
      size_t offset;
      uint8_t width;
      size_t index;
      size_t relative_to;
      bool is_signed;
    };

    std::vector<uint8_t>& code_;
    std::vector<std::optional<RelocationValue>> relocatables_;
    std::vector<Relocation> relocations_;
  };

  /**
   * Opaque handle to a relocatable value.
   */
  struct Generator::Relocatable
  {
  private:
    Relocatable(size_t index) : index(index) {}
    size_t index;
    friend Generator;
  };

  /**
   * Wrapper around Relocatable, used for offsets within the program.
   */
  struct Label
  {
    explicit Label(Generator::Relocatable relocatable)
    : relocatable(relocatable)
    {}
    Generator::Relocatable relocatable;

    operator Generator::Relocatable() const
    {
      return relocatable;
    }
  };

  /**
   * Wrapper around Relocatable, used for descriptor indices.
   */
  struct Descriptor
  {
    explicit Descriptor(Generator::Relocatable relocatable)
    : relocatable(relocatable)
    {}
    Generator::Relocatable relocatable;

    operator Generator::Relocatable() const
    {
      return relocatable;
    }
  };

  struct CalleeRegister
  {
    CalleeRegister(
      size_t callspace,
      Generator::Relocatable frame_size,
      bytecode::Register reg)
    : callspace(callspace), frame_size(frame_size), reg(reg)
    {
      if (reg.value >= callspace)
        throw std::logic_error(
          "Cannot access callee argument beyond call space");
    }

    size_t callspace;
    Generator::Relocatable frame_size;
    bytecode::Register reg;
  };

  template<typename T>
  struct Generator::emit_helper<T, std::enable_if_t<std::is_integral_v<T>>>
  {
    template<typename U>
    static void write(Generator* gen, size_t opcode_start, U value)
    {
      static_assert(
        std::is_same_v<T, U>, "Attempting to convert types implicitely");
      gen->write<T>(value);
    }
  };

  template<typename T>
  struct Generator::emit_helper<T, std::enable_if_t<detail::is_enum_class_v<T>>>
  {
    static void write(Generator* gen, size_t opcode_start, T value)
    {
      using wire_type = std::underlying_type_t<T>;
      gen->write<wire_type>(static_cast<wire_type>(value));
    }
  };

  template<typename T>
  struct Generator::emit_helper<T, std::enable_if_t<bytecode::is_wrapper_v<T>>>
  {
    static void write(Generator* gen, size_t opcode_start, T value)
    {
      gen->write<typename T::underlying_type>(value.value);
    }
  };

  template<>
  struct Generator::emit_helper<bytecode::Register>
  {
    static void
    write(Generator* gen, size_t opcode_start, bytecode::Register value)
    {
      gen->u8(value.value);
    }

    static void
    write(Generator* gen, size_t opcode_start, const CalleeRegister& value)
    {
      gen->u8(value.frame_size, value.callspace - value.reg.value);
    }
  };

  template<>
  struct Generator::emit_helper<bytecode::DescriptorIdx>
  {
    static void write(Generator* gen, size_t opcode_start, Descriptor value)
    {
      gen->u32(value);
    }
  };

  template<>
  struct Generator::emit_helper<std::string_view>
  {
    static void
    write(Generator* gen, size_t opcode_start, std::string_view value)
    {
      gen->str(value);
    }
  };

  template<>
  struct Generator::emit_helper<bytecode::RegisterSpan>
  {
    static void
    write(Generator* gen, size_t opcode_start, bytecode::RegisterSpan value)
    {
      size_t size = value.size();
      gen->u8(truncate<uint8_t>(size));
      for (bytecode::Register reg : value)
      {
        gen->u8(reg.value);
      }
    }
  };

  template<>
  struct Generator::emit_helper<bytecode::RelativeOffset>
  {
    static void write(Generator* gen, size_t opcode_start, Label value)
    {
      gen->s16(value, opcode_start);
    }
  };

  template<>
  struct Generator::emit_helper<bytecode::AbsoluteOffset>
  {
    static void write(Generator* gen, size_t opcode_start, Label value)
    {
      gen->u32(value);
    }
  };
}
