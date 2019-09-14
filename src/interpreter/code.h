// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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
    uint16_t u32(size_t& ip) const
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
      entrypoint_ = u32(ip);

      uint16_t descriptors_count = u16(ip);
      for (uint16_t i = 0; i < descriptors_count; i++)
      {
        descriptors_.push_back(load_descriptor(ip));
      }
    }

    const std::vector<std::unique_ptr<const VMDescriptor>>& descriptors()
    {
      return descriptors_;
    }

    size_t entrypoint() const
    {
      return entrypoint_;
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

  private:
    const std::vector<uint8_t> data_;
    std::vector<std::unique_ptr<const VMDescriptor>> descriptors_;
    size_t entrypoint_;

    std::unique_ptr<VMDescriptor> load_descriptor(size_t& ip)
    {
      std::string_view name = str(ip);
      uint16_t method_slots = u16(ip);
      uint16_t method_count = u16(ip);
      uint16_t field_slots = u16(ip);
      uint16_t field_count = u16(ip);
      uint32_t finaliser_slot = u32(ip);

      auto descriptor = std::make_unique<VMDescriptor>(
        name, method_slots, field_slots, field_count, finaliser_slot);

      for (uint8_t i = 0; i < method_count; i++)
      {
        SelectorIdx index = selector(ip);
        uint32_t offset = u32(ip);
        assert(index < method_slots);
        descriptor->methods[index] = offset;
      }
      for (uint8_t i = 0; i < field_count; i++)
      {
        SelectorIdx index = selector(ip);
        assert(index < field_slots);
        descriptor->fields[index] = i;
      }

      return descriptor;
    }

    template<typename T, typename = void>
    struct load_helper;

    template<typename T>
    struct load_helper<T, std::enable_if_t<std::is_integral_v<T>>>
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
    struct load_helper<T, std::enable_if_t<std::is_enum_v<T>>>
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

    template<typename Dummy>
    struct load_helper<std::string_view, Dummy>
    {
      static std::string_view load(const Code& code, size_t& ip)
      {
        uint16_t size = code.u16(ip);
        code.check(ip, size);
        std::string_view s(
          reinterpret_cast<const char*>(&code.data_[ip]), size);
        ip += size;
        return s;
      }
    };
  };

  template<>
  inline bytecode::Register Code::load<bytecode::Register>(size_t& ip) const
  {
    return bytecode::Register(load<uint8_t>(ip));
  }
}
