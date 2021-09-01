// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/bytecode.h"
#include "interpreter/value.h"

#include <unordered_set>
#include <verona.h>

namespace verona::interpreter
{
  struct VMDescriptor : public rt::Descriptor
  {
    VMDescriptor(
      bytecode::DescriptorIdx index,
      std::string_view name,
      size_t method_slots,
      size_t field_slots,
      size_t field_count,
      uint32_t finaliser_ip);

    const bytecode::DescriptorIdx index;
    const std::string name;
    const size_t field_count;
    std::unique_ptr<uint32_t[]> fields;
    std::unique_ptr<uint32_t[]> methods;
    std::unordered_set<bytecode::DescriptorIdx> subtypes;
    const uint32_t finaliser_ip;
  };

  struct VMObject : public rt::Object
  {
    /**
     * `region` should be the region which contains this object.
     *
     * If the object is in a new region, nullptr should be passed instead.
     */
    explicit VMObject(VMObject* region, const VMDescriptor* desc);

    std::unique_ptr<FieldValue[]> fields;

    const VMDescriptor* descriptor() const
    {
      return static_cast<const VMDescriptor*>(rt::Object::get_descriptor());
    }

    VMObject* region();

    static void trace_fn(const rt::Object* base_object, rt::ObjectStack& stack);
    static void finaliser_fn(
      rt::Object* base_object,
      rt::Object* region,
      rt::ObjectStack& sub_regions);
    static void collect_iso_fields(
      rt::Object* base_object,
      rt::Object* region,
      rt::ObjectStack& sub_regions);
    static void destructor_fn(rt::Object* base_object);

  private:
    VMObject* parent_;
  };

  struct VMCown : public rt::VCown<VMCown>
  {
    // This is the descriptor for cown[T], not for T.
    // It is used to dispatch methods on the cown itself.
    const VMDescriptor* descriptor;
    VMObject* contents;

    /**
     * contents should be a region entrypoint. VMCown will take ownership of it.
     */
    explicit VMCown(const VMDescriptor* descriptor, VMObject* contents)
    : descriptor(descriptor), contents(contents)
    {
      assert((contents == nullptr) || contents->debug_is_iso());
    }

    /**
     * This is for promises., the cown should be initially unscheduled.
     */
    explicit VMCown(const VMDescriptor* descriptor)
    : descriptor(descriptor), contents(nullptr)
    {
      wake();
    }

    void schedule()
    {
      rt::VCown<VMCown>::schedule();
    }

    void trace(rt::ObjectStack& stack)
    {
      if (contents != nullptr)
        stack.push(contents);
    }
  };
}
