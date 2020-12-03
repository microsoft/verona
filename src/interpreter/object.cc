// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "interpreter/object.h"

#include "vm.h"

#include <fmt/ostream.h>

namespace verona::interpreter
{
  VMDescriptor::VMDescriptor(
    bytecode::DescriptorIdx index,
    std::string_view name,
    size_t method_slots,
    size_t field_slots,
    size_t field_count,
    uint32_t finaliser_ip)
  : index(index),
    name(name),
    methods(std::make_unique<uint32_t[]>(method_slots)),
    fields(std::make_unique<uint32_t[]>(field_slots)),
    field_count(field_count),
    finaliser_ip(finaliser_ip)
  {
    rt::Descriptor::size = rt::vsizeof<VMObject>;
    rt::Descriptor::trace = VMObject::trace_fn;

    // Try to be on the trivial ring as much as possible. This requires the
    // following three methods to be null.
    //
    // If there are no fields, then VMObject's destructor is a no-op and we can
    // skip it. In that case, there are also obviously no iso fields.
    rt::Descriptor::destructor =
      field_count > 0 ? VMObject::destructor_fn : nullptr;
    // In the VM object we always need a finaliser in case it has an iso field.
    // The finaliser will collect the iso references that will be deallocated
    // at the end of this phase.
    // TODO: Can be optimised if we look at the types of all the fields
    rt::Descriptor::finaliser =
      finaliser_ip > 0 ? VMObject::finaliser_fn : VMObject::collect_iso_fields;

    // Make sure `subtypes` is reflexive. This simplifies code that uses the
    // set, by removing special casing.
    subtypes.insert(index);
  }

  VMObject::VMObject(VMObject* region, const VMDescriptor* desc)
  : Object(), parent_(region)
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

  void VMObject::trace_fn(const rt::Object* base_object, rt::ObjectStack& stack)
  {
    const VMObject* object = static_cast<const VMObject*>(base_object);
    const VMDescriptor* descriptor = object->descriptor();

    for (size_t i = 0; i < descriptor->field_count; i++)
    {
      object->fields[i].trace(stack);
    }
  }

  void VMObject::finaliser_fn(
    rt::Object* base_object, rt::Object* region, rt::ObjectStack& sub_regions)
  {
    VMObject* object = static_cast<VMObject*>(base_object);

    VM::execute_finaliser(object);

    collect_iso_fields(base_object, region, sub_regions);
  }

  void VMObject::collect_iso_fields(
    rt::Object* base_object, rt::Object* region, rt::ObjectStack& sub_regions)
  {
    // The interpreter doesn't need the region, as the FieldValue type contains
    // the ISO information in its fat pointers.
    UNUSED(region);

    VMObject* object = static_cast<VMObject*>(base_object);
    const VMDescriptor* descriptor = object->descriptor();

    for (size_t i = 0; i < descriptor->field_count; i++)
    {
      object->fields[i].add_isos(sub_regions);
    }
  }

  void VMObject::destructor_fn(rt::Object* base_object)
  {
    VMObject* object = static_cast<VMObject*>(base_object);
    (object)->~VMObject();
  }
}
