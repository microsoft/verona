// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/bytecode.h"
#include "interpreter/object.h"

#include <fmt/ostream.h>
#include <optional>
#include <verona.h>

namespace verona::interpreter
{
  using bytecode::CodePtr;
  using bytecode::DescriptorIdx;
  using bytecode::FunctionHeader;
  using bytecode::Opcode;
  using bytecode::SelectorIdx;

  /**
   * Programs have a few special descriptors and selectors which the VM needs to
   * be aware of. Their value is encoded in the program's header, and is be
   * loaded into this struct.
   */
  struct SpecialDescriptors
  {
    const VMDescriptor* main;
    SelectorIdx main_selector;
    const VMDescriptor* u64;
  };

  class Code
  {
  public:
    void check(size_t ip, size_t len) const
    {
      if ((ip + len) > data_.size())
      {
        std::stringstream s;
        s << "Instruction overflow " << ip << " " << len;
        throw std::logic_error(s.str());
      }
    }

    template<typename T>
    T load(size_t& ip) const
    {
      return load_helper<T>::load(*this, ip);
    }

    /**
     * Load all operands of an opcode as a tuple.
     *
     * The return value is a tuple whose elements match the Operands specified
     * in this opcode's OpcodeSpec.
     */
    template<Opcode opcode>
    auto load_operands(size_t& ip) const
    {
      return load_operands_inner<opcode>(
        typename bytecode::OpcodeSpec<opcode>::Operands(), ip);
    }

    /**
     * Helper function used to implement load_operands.
     *
     * Splitting this in two function, and passing an (empty) OpcodeOperands
     * value allows the Args... parameter pack to be deduced.
     */
    template<Opcode opcode, typename... Args>
    std::tuple<Args...>
    load_operands_inner(bytecode::OpcodeOperands<Args...>, size_t& ip) const
    {
      return {load<Args>(ip)...};
    }

    uint8_t u8(size_t& ip) const
    {
      return load<uint8_t>(ip);
    }
    uint16_t u16(size_t& ip) const
    {
      return load<uint16_t>(ip);
    }
    int16_t s16(size_t& ip) const
    {
      return load<int16_t>(ip);
    }

    uint32_t u32(size_t& ip) const
    {
      return load<uint32_t>(ip);
    }

    uint64_t u64(size_t& ip) const
    {
      return load<uint64_t>(ip);
    }

    DescriptorIdx descriptor(size_t& ip) const
    {
      return load<DescriptorIdx>(ip);
    }

    SelectorIdx selector(size_t& ip) const
    {
      return load<SelectorIdx>(ip);
    }

    Opcode opcode(size_t& ip) const
    {
      return load<Opcode>(ip);
    }

    std::string_view str(size_t& ip) const
    {
      return load<std::string_view>(ip);
    }

    size_t relative_label(size_t& ip) const
    {
      size_t position = ip;
      ptrdiff_t offset = s16(ip);
      return position + offset;
    }

    FunctionHeader function_header(size_t& ip) const
    {
      FunctionHeader header;
      header.name = str(ip);
      header.argc = u8(ip);
      header.retc = u8(ip);
      header.locals = u8(ip);
      header.size = u32(ip);
      return header;
    }

    Code(std::vector<uint8_t> code) : data_(std::move(code))
    {
      size_t ip = 0;

      check_verona_nums(ip);

      uint32_t descriptors_count = u32(ip);
      for (uint32_t i = 0; i < descriptors_count; i++)
      {
        descriptors_.push_back(load_descriptor(ip));
      }

      special_descriptors_.main = get_descriptor(load<DescriptorIdx>(ip));
      special_descriptors_.main_selector = load<SelectorIdx>(ip);
      special_descriptors_.u64 =
        get_optional_descriptor(load<DescriptorIdx>(ip));
    }

    const std::vector<std::unique_ptr<const VMDescriptor>>& descriptors()
    {
      return descriptors_;
    }

    const SpecialDescriptors& special_descriptors() const
    {
      return special_descriptors_;
    }

    size_t entrypoint() const
    {
      SelectorIdx selector = special_descriptors_.main_selector;
      return special_descriptors_.main->methods[selector];
    }

    const VMDescriptor* get_descriptor(DescriptorIdx desc) const
    {
      if (desc >= descriptors_.size())
      {
        std::stringstream s;
        s << "Invalid descriptor id " << (int)desc;
        throw std::logic_error(s.str());
      }
      return descriptors_.at(desc).get();
    }

    const VMDescriptor* get_optional_descriptor(DescriptorIdx desc) const
    {
      if (desc == bytecode::INVALID_DESCRIPTOR)
        return nullptr;
      else
        return get_descriptor(desc);
    }

  private:
    const std::vector<uint8_t> data_;
    std::vector<std::unique_ptr<const VMDescriptor>> descriptors_;

    SpecialDescriptors special_descriptors_;

    void check_verona_nums(size_t& ip)
    {
      uint32_t nums = u32(ip);
      if (nums != bytecode::MAGIC_NUMBER)
      {
        throw std::logic_error{"Invalid magic number, not recognized"};
      }
    }

    std::unique_ptr<VMDescriptor> load_descriptor(size_t& ip)
    {
      std::string_view name = str(ip);
      uint32_t method_slots = u32(ip);
      uint32_t method_count = u32(ip);
      uint32_t field_slots = u32(ip);
      uint32_t field_count = u32(ip);
      uint32_t finaliser_ip = u32(ip);

      auto descriptor = std::make_unique<VMDescriptor>(
        name, method_slots, field_slots, field_count, finaliser_ip);

      for (uint32_t i = 0; i < method_count; i++)
      {
        SelectorIdx index = selector(ip);
        uint32_t offset = u32(ip);
        assert(index < method_slots);
        descriptor->methods[index] = offset;
      }
      for (uint32_t i = 0; i < field_count; i++)
      {
        SelectorIdx index = selector(ip);
        assert(index < field_slots);
        descriptor->fields[index] = i;
      }

      return descriptor;
    }

    /**
     * Helper class that allows us to override the behaviour based on the type
     * of the operand we want to load.
     *
     * Different specializations are provided below for various types T. Every
     * specialization provides a `static T load(const Code& code, size_t& ip)`
     * method.
     */
    template<typename T, typename = void>
    struct load_helper;
  };

  template<typename T>
  struct Code::load_helper<T, std::enable_if_t<std::is_integral_v<T>>>
  {
    static T load(const Code& code, size_t& ip)
    {
      code.check(ip, sizeof(T));
      uint64_t bits = 0;

      for (size_t i = 0; i < (sizeof(T) * 8); i += 8)
      {
        bits |= code.data_[ip++] << i;
      }

      return (T)bits;
    }
  };

  template<typename T>
  struct Code::load_helper<T, std::enable_if_t<std::is_enum_v<T>>>
  {
    static T load(const Code& code, size_t& ip)
    {
      typedef std::underlying_type_t<T> wire_type;
      wire_type value = code.load<wire_type>(ip);
      if (value > static_cast<wire_type>(T::maximum_value))
      {
        throw std::logic_error(
          fmt::format("Invalid value {:d} for {}", value, typeid(T).name()));
      }
      return static_cast<T>(value);
    }
  };

  template<>
  struct Code::load_helper<std::string_view>
  {
    static std::string_view load(const Code& code, size_t& ip)
    {
      uint16_t size = code.u16(ip);
      code.check(ip, size);
      std::string_view s(reinterpret_cast<const char*>(&code.data_[ip]), size);
      ip += size;
      return s;
    }
  };

  template<>
  struct Code::load_helper<bytecode::RegisterSpan>
  {
    static bytecode::RegisterSpan load(const Code& code, size_t& ip)
    {
      uint8_t size = code.u8(ip);
      code.check(ip, size);
      const bytecode::Register* data =
        reinterpret_cast<const bytecode::Register*>(&code.data_[ip]);
      ip += size;
      return bytecode::RegisterSpan(data, size);
    }
  };

  template<>
  struct Code::load_helper<bytecode::Register>
  {
    static bytecode::Register load(const Code& code, size_t& ip)
    {
      return bytecode::Register(code.load<uint8_t>(ip));
    }
  };
}
