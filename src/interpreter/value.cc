// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "interpreter/value.h"

#include "ds/helpers.h"
#include "interpreter/object.h"

namespace verona::interpreter
{
  Value Value::u64(uint64_t value)
  {
    Value v;
    v.tag = U64;
    v.inner.u64 = value;
    return v;
  }

  Value Value::string(std::string value)
  {
    Value v;
    v.tag = STRING;
    v.inner.string_ptr = new std::string(std::move(value));
    return v;
  }

  Value Value::string(std::string_view value)
  {
    return Value::string(std::string(value));
  }

  Value Value::iso(VMObject* object)
  {
    assert(object->debug_is_iso());
    Value v;
    v.tag = ISO;
    v.inner.object = object;
    return v;
  }

  Value Value::mut(VMObject* object)
  {
    assert(object->debug_is_iso() || object->debug_is_mutable());
    Value v;
    v.tag = MUT;
    v.inner.object = object;
    return v;
  }

  Value Value::imm(VMObject* object)
  {
    assert(object->debug_is_immutable());
    Value v;
    v.tag = IMM;
    v.inner.object = object;
    return v;
  }

  Value Value::cown(VMCown* cown)
  {
    Value v;
    v.tag = COWN;
    v.inner.cown = cown;
    return v;
  }

  Value Value::unowned_cown(VMCown* cown)
  {
    Value v;
    v.tag = COWN_UNOWNED;
    v.inner.cown = cown;
    return v;
  }

  Value Value::descriptor(const VMDescriptor* descriptor)
  {
    Value v;
    v.tag = DESCRIPTOR;
    v.inner.descriptor = descriptor;
    return v;
  }

  Value::Value(Value&& other)
  {
    this->tag = other.tag;
    this->inner = other.inner;
    other.tag = UNINIT;
  }

  void Value::overwrite(rt::Alloc* alloc, Value&& other)
  {
    std::swap(this->tag, other.tag);
    std::swap(this->inner, other.inner);
    other.clear(alloc);
  }

  Value::~Value()
  {
    if (tag != UNINIT)
    {
      std::cerr << "Dropped an initialized Value" << std::endl;
      abort();
    }
  }

  void Value::clear(rt::Alloc* alloc)
  {
    switch (tag)
    {
      case COWN:
        rt::Cown::release(alloc, inner.cown);
        break;

      case ISO:
        rt::Region::release(alloc, inner.object);
        break;

      case IMM:
        rt::Immutable::release(alloc, inner.object);
        break;

      case STRING:
        delete inner.string_ptr;
        break;

      case MUT:
      case UNINIT:
      case U64:
      case DESCRIPTOR:
      case COWN_UNOWNED:
        break;
    }
    tag = UNINIT;
  }

  VMCown* Value::consume_cown()
  {
    switch (tag)
    {
      case COWN:
        tag = UNINIT;
        return inner.cown;
      default:
        abort();
    }
  }

  Value Value::as_unowned_cown() const
  {
    switch (tag)
    {
      case COWN:
        return Value::unowned_cown(inner.cown);

      default:
        abort();
    }
  }

  Value Value::cown_body() const
  {
    switch (tag)
    {
      case COWN:
      case COWN_UNOWNED:
        return Value::mut(inner.cown->contents);
      default:
        abort();
    }
  }

  Value Value::maybe_consume()
  {
    switch (tag)
    {
      case UNINIT:
        return Value();

      case U64:
        return Value::u64(inner.u64);

      case STRING:
        return Value::string(inner.string());

      case DESCRIPTOR:
        return Value::descriptor(inner.descriptor);

      case COWN:
        rt::Cown::acquire(inner.cown);
        return Value::cown(inner.cown);

      case COWN_UNOWNED:
        abort();

      case ISO:
        // Mark the Value as empty, since we are transferring ownership of the
        // region out of it.
        tag = UNINIT;
        return Value::iso(inner.object);

      case MUT:
        return Value::mut(inner.object);

      case IMM:
        rt::Immutable::acquire(inner.object);
        return Value::imm(inner.object);

        EXHAUSTIVE_SWITCH
    }
  }

  VMObject* Value::consume_iso()
  {
    assert(tag == ISO);
    tag = UNINIT;
    return inner.object;
  }

  Value FieldValue::read(Value::Tag parent)
  {
    assert(
      parent == Value::ISO || parent == Value::MUT || parent == Value::IMM);
    switch (tag)
    {
      case Value::UNINIT:
        return Value();

      case Value::U64:
        return Value::u64(inner.u64);

      case Value::STRING:
        return Value::string(inner.string());

      case Value::DESCRIPTOR:
        return Value::descriptor(inner.descriptor);

      case Value::ISO:
        if (parent == Value::IMM)
        {
          rt::Immutable::acquire(inner.object);
          return Value::imm(inner.object);
        }

        // We return a MUT value, even if this is an ISO field.
        // FieldValue::exchange must be used to extract the field as ISO.
        return Value::mut(inner.object);

      case Value::MUT:
        if (parent == Value::IMM)
        {
          rt::Immutable::acquire(inner.object);
          return Value::imm(inner.object);
        }

        return Value::mut(inner.object);

      case Value::IMM:
        rt::Immutable::acquire(inner.object);
        return Value::imm(inner.object);

      case Value::COWN:
        rt::Cown::acquire(inner.cown);
        return Value::cown(inner.cown);

      case Value::COWN_UNOWNED:
        // Cannot be used in the heap.  Only used in messages
        abort();

        EXHAUSTIVE_SWITCH
    }
  }

  Value
  FieldValue::exchange(rt::Alloc* alloc, rt::Object* region, Value&& value)
  {
    switch (value.tag)
    {
      case Value::IMM:
        assert(value.inner.object->debug_is_immutable());
        // TODO(region): For now, only allow inserting into trace regions.
        assert(rt::RegionTrace::is_trace_region(rt::Region::get(region)));
        rt::RegionTrace::insert<rt::YesTransfer>(
          alloc, region, value.inner.object);
        break;
      case Value::COWN:
        // TODO(region): For now, only allow inserting into trace regions.
        assert(rt::RegionTrace::is_trace_region(rt::Region::get(region)));
        rt::RegionTrace::insert<rt::YesTransfer>(
          alloc, region, value.inner.cown);
        break;
      default:
        break;
    }

    switch (tag)
    {
      case Value::IMM:
        assert(inner.object->debug_is_immutable());
        rt::Immutable::acquire(inner.object);
        break;

      case Value::COWN:
        rt::Cown::acquire(inner.cown);
        break;

      default:
        break;
    }

    Value result;
    result.tag = this->tag;
    result.inner = this->inner;
    this->tag = value.tag;
    this->inner = value.inner;
    value.tag = Value::UNINIT;

    return result;
  }

  void FieldValue::trace(rt::ObjectStack& stack) const
  {
    switch (tag)
    {
      case Value::ISO:
      case Value::MUT:
      case Value::IMM:
        stack.push(inner.object);
        break;

      case Value::COWN:
        stack.push(inner.cown);
        break;

      case Value::UNINIT:
      case Value::U64:
      case Value::STRING:
      case Value::DESCRIPTOR:
        break;

      case Value::COWN_UNOWNED:
        // Cannot be part of the heap.
        abort();

        EXHAUSTIVE_SWITCH
    }
  }

  void FieldValue::add_isos(rt::ObjectStack& stack) const
  {
    switch (tag)
    {
      case Value::ISO:
        stack.push(inner.object);
        break;

      case Value::COWN:
      case Value::MUT:
      case Value::IMM:
      case Value::UNINIT:
      case Value::U64:
      case Value::STRING:
      case Value::DESCRIPTOR:
        break;

      case Value::COWN_UNOWNED:
        // Cannot be part of the heap.
        abort();

        EXHAUSTIVE_SWITCH
    }
  }

  FieldValue::~FieldValue()
  {
    switch (tag)
    {
      case Value::STRING:
        delete inner.string_ptr;
        break;

      case Value::ISO:
      case Value::MUT:
      case Value::IMM:
      case Value::COWN:
        // These are handled by the GC.
        break;

      case Value::DESCRIPTOR:
      case Value::U64:
      case Value::UNINIT:
        break;

      case Value::COWN_UNOWNED:
        // Cannot be part of the heap.
        abort();

        EXHAUSTIVE_SWITCH
    }
  }
}
