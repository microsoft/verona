// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/bytecode.h"

#include <functional>
#include <iostream>
#include <optional>
#include <vector>

namespace verona::compiler
{
  struct Label;
  struct Descriptor;

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

    void reg(bytecode::Register reg);
    void opcode(bytecode::Opcode opcode);
    void selector(bytecode::SelectorIdx index);
    void descriptor(Descriptor descriptor);

    /**
     * Emit a list of registers, prefixed by an 8-bit length.
     */
    void reglist(bytecode::RegisterSpan regs);

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
     * Write a child-relative register index.
     *
     * This depends on the call space of the child function, and the size of the
     * current frame. Since the latter is not usually known until the function
     * is fully generated, a relocatable value is used instead.
     */
    void child_register(
      size_t child_callspace, Relocatable frame_size, bytecode::Register reg);

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

  private:
    /**
     * Write an integer value in little endian format.
     *
     * The common_type_t disables template parameter deduction, forcing the
     * caller the specify the integer type explicitly.
     */
    template<typename T>
    void write(std::common_type_t<T> value);

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
}
