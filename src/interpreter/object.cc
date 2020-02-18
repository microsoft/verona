// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "interpreter/object.h"

#include "vm.h"

#include <fmt/ostream.h>

namespace verona::interpreter
{
  VMDescriptor::VMDescriptor(
    std::string_view name,
    size_t method_slots,
    size_t field_slots,
    size_t field_count,
    uint32_t finaliser_ip)
  : name(name),
    methods(std::make_unique<uint32_t[]>(method_slots)),
    fields(std::make_unique<uint32_t[]>(field_slots)),
    field_count(field_count),
    finaliser_ip(finaliser_ip)
  {
    rt::Descriptor::size = sizeof(VMObject);
    rt::Descriptor::trace = VMObject::trace_fn;

    // Try to be on the trivial ring as much as possible. This requires the
    // following three methods to be null.
    //
    // If there are no fields, then VMObject's destructor is a no-op and we can
    // skip it. In that case, there are also obviously no iso fields.
    rt::Descriptor::destructor =
      field_count > 0 ? VMObject::destructor_fn : nullptr;
    rt::Descriptor::trace_possibly_iso =
      field_count > 0 ? VMObject::trace_fn : nullptr;
    rt::Descriptor::finaliser =
      finaliser_ip > 0 ? VMObject::finaliser_fn : nullptr;
  }

  VMObject::VMObject(VMObject* region) : parent_(region)
  {
    if (descriptor()->field_count > 0)
      fields = std::make_unique<FieldValue[]>(descriptor()->field_count);
    else
      fields = nullptr;
  }

  VMObject* VMObject::region()
  {
    if (parent_ == nullptr)
    {
      return this;
    }
    else
    {
      parent_ = parent_->region();
      return parent_;
    }
  }

  void VMObject::trace_fn(const rt::Object* base_object, rt::ObjectStack* stack)
  {
    const VMObject* object = static_cast<const VMObject*>(base_object);
    const VMDescriptor* descriptor = object->descriptor();

    for (size_t i = 0; i < descriptor->field_count; i++)
    {
      object->fields[i].trace(stack);
    }
  }

  void VMObject::finaliser_fn(rt::Object* base_object)
  {
    VMObject* object = static_cast<VMObject*>(base_object);
    VM::execute_finaliser(object);
  }

  void VMObject::destructor_fn(rt::Object* base_object)
  {
    VMObject* object = static_cast<VMObject*>(base_object);
    (object)->~VMObject();
  }
}
