// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/generator.h"

#include "ds/helpers.h"

#include <cassert>

namespace verona::compiler
{
  void Generator::u8(uint8_t value)
  {
    write<uint8_t>(value);
  }

  void Generator::u16(uint16_t value)
  {
    write<uint16_t>(value);
  }

  void Generator::u32(uint32_t value)
  {
    write<uint32_t>(value);
  }

  void Generator::u64(uint64_t value)
  {
    write<uint64_t>(value);
  }

  void Generator::str(std::string_view s)
  {
    u16(truncate<uint16_t>(s.size()));
    code_.insert(code_.end(), s.data(), s.data() + s.size());
  }

  void Generator::reg(bytecode::Register reg)
  {
    u8(reg.index);
  }

  void Generator::reglist(bytecode::RegisterSpan regs)
  {
    size_t size = regs.size();
    u8(truncate<uint8_t>(size));
    for (bytecode::Register r : regs)
    {
      reg(r);
    }
  }

  void Generator::opcode(bytecode::Opcode opcode)
  {
    u8((uint8_t)opcode);
  }

  void Generator::selector(bytecode::SelectorIdx index)
  {
    u32(index);
  }

  void Generator::descriptor(Descriptor descriptor)
  {
    u32(descriptor);
  }

  void Generator::u8(Relocatable relocatable, size_t relative_to)
  {
    add_relocation(current_offset(), 1, relocatable, relative_to, false);
    u8(0);
  }

  void Generator::u16(Relocatable relocatable, size_t relative_to)
  {
    add_relocation(current_offset(), 2, relocatable, relative_to, false);
    u16(0);
  }

  void Generator::s16(Relocatable relocatable, size_t relative_to)
  {
    add_relocation(current_offset(), 2, relocatable, relative_to, true);
    u16(0);
  }

  void Generator::u32(Relocatable relocatable, size_t relative_to)
  {
    add_relocation(current_offset(), 4, relocatable, relative_to, false);
    u32(0);
  }

  void Generator::u64(Relocatable relocatable, size_t relative_to)
  {
    add_relocation(current_offset(), 8, relocatable, relative_to, false);
    u64(0);
  }

  void Generator::child_register(
    size_t child_callspace, Relocatable frame_size, bytecode::Register reg)
  {
    if (reg.index >= child_callspace)
      throw std::logic_error("Cannot access child argument beyond call space");

    u8(frame_size, child_callspace - reg.index);
  }

  template<typename T>
  void Generator::write(std::common_type_t<T> value)
  {
    static_assert(std::is_integral_v<T>);

    size_t offset = code_.size();
    code_.reserve(offset + sizeof(T));
    for (size_t i = 0; i < sizeof(T) * 8; i += 8)
    {
      code_.push_back((value >> i) & 0xff);
    }
  }

  void Generator::finish()
  {
    for (const auto& rel : relocations_)
    {
      const std::optional<RelocationValue>& slot = relocatables_.at(rel.index);
      if (!slot.has_value())
      {
        throw std::logic_error("Undefined label");
      }

      assert(rel.offset + rel.width <= code_.size());

      // Checks after making the relative value that it is a valid signed or
      // unsigned value of the correct size.  That is it hasn't overflowed
      // the n-bit value.
      uint64_t value = *slot - rel.relative_to;
      if (rel.width != 8)
      {
        size_t shift = (8 - rel.width) * 8;
        if (rel.is_signed)
        {
          // Check that all the bits beyond the value when restricted to
          // its size, are the same as the top bit.
          if (value != (((int64_t)value) << shift) >> shift)
            abort();
        }
        else
        {
          // Check top bits are all zero.
          if (value != (value << shift) >> shift)
            abort();
        }
      }

      // Write back the relocated value
      for (int i = 0; i < rel.width; i++)
      {
        code_.at(rel.offset + i) = (value >> (i * 8)) & 0xff;
      }
    }
  }

  void Generator::add_relocation(
    size_t offset,
    uint8_t width,
    Relocatable relocatable,
    size_t relative_to,
    bool is_signed)
  {
    assert(width <= 8);
    relocations_.push_back(
      {offset, width, relocatable.index, relative_to, is_signed});
  }

  Generator::Relocatable Generator::create_relocatable()
  {
    size_t index = relocatables_.size();
    relocatables_.push_back(std::nullopt);
    return Relocatable(index);
  }

  Label Generator::create_label()
  {
    return Label(create_relocatable());
  }

  Descriptor Generator::create_descriptor()
  {
    return Descriptor(create_relocatable());
  }

  void
  Generator::define_relocatable(Relocatable relocatable, RelocationValue value)
  {
    std::optional<RelocationValue>& slot = relocatables_.at(relocatable.index);
    if (slot.has_value())
      throw std::logic_error("Relocatable already has a value");

    slot = value;
  }

  void Generator::define_label(Label label)
  {
    define_relocatable(label, current_offset());
  }
}
