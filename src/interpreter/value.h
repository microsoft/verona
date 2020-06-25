// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/bytecode.h"

#include <fmt/format.h>
#include <verona.h>

namespace verona::interpreter
{
  struct VMDescriptor;
  struct VMObject;
  struct VMCown;

  /**
   * Tagged Verona value, which handles ownership of objects and reference
   * counts.
   *
   * Used for stack variables, and temporaries through out the VM
   * implementation.
   *
   * Because releasing ownership may require access to the local allocator, a
   * Value must explicitly be cleared before destruction, by calling
   * `clear(Alloc*)`. Failing to do so will result in an abort.
   */
  struct Value
  {
    enum class Tag
    {
      UNINIT,

      ISO,
      MUT, // Represents both mut(x) and "borrowed iso" iso(x)
      IMM,

      DESCRIPTOR,
      U64,
      COWN,
      COWN_UNOWNED,

      STRING,
    };

    union Inner
    {
      // Used by the ISO, MUT and IMM variants.
      VMObject* object;
      VMCown* cown;
      const VMDescriptor* descriptor;
      uint64_t u64;
      std::string* string_ptr;

      std::string& string()
      {
        return *string_ptr;
      }

      const std::string& string() const
      {
        return *string_ptr;
      }
    };

    Tag tag;
    Inner inner;

    Value() : tag(Tag::UNINIT) {}

    static Value u64(uint64_t value);
    static Value string(std::string value);
    static Value string(std::string_view value);

    // Takes ownership of the region.
    static Value iso(VMObject* object);
    static Value mut(VMObject* object);
    // Takes ownership of the reference count.
    static Value imm(VMObject* object);
    // Takes ownership of the reference count.
    static Value cown(VMCown* cown);

    static Value descriptor(const VMDescriptor* descriptor);

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    Value& operator=(Value&& other) = delete;

    Value(Value&& other);
    ~Value();

    /**
     * Clear the Value, making it UNINIT.
     *
     * It will release any ownership of regions or reference counts it may have.
     */
    void clear(rt::Alloc* alloc);

    /**
     * Replace the contents of the Value.
     *
     * This moves the contents of `other` into this Value, and releases the old
     * one. It's essentially a move assignment operator, but with access to the
     * memory allocator.
     */
    void overwrite(rt::Alloc* alloc, Value&& other);

    /**
     * Get a copy of this Value, by maybe consuming it.
     *
     * If the value is an ISO, the ownership of the region is transferred out
     * and the value is cleared to UNINIT. Otherwise the old value is preserved.
     *
     * If the value is reference counted, it is incremented and the return value
     * owns the new reference count.
     */
    Value maybe_consume();

    /**
     * Consume an ISO value, extracting the underlying VMObject*. Ownership is
     * transferred with the return value.
     *
     * This Value is cleared to UNINIT.
     */
    VMObject* consume_iso();

    /**
     * Consume a COWN value, making it unowned. Ownership is released to the
     * caller.  But the cown is still intact.  This is used when ownership of a
     * cown is passed to the runtime in a multimessage.
     */
    void consume_cown();

    /**
     * On an unowned cown, this converts it to a reference to the underlying
     * object.
     */
    void switch_to_cown_body();

    Inner* operator->()
    {
      return &inner;
    }
    const Inner* operator->() const
    {
      return &inner;
    }

    void trace(rt::ObjectStack& stack) const;

    static constexpr Tag UNINIT = Tag::UNINIT;
    static constexpr Tag ISO = Tag::ISO;
    static constexpr Tag MUT = Tag::MUT;
    static constexpr Tag IMM = Tag::IMM;
    static constexpr Tag U64 = Tag::U64;
    static constexpr Tag DESCRIPTOR = Tag::DESCRIPTOR;
    static constexpr Tag COWN = Tag::COWN;
    static constexpr Tag COWN_UNOWNED = Tag::COWN_UNOWNED;
    static constexpr Tag STRING = Tag::STRING;
  };

  /**
   * Alternative to Value used for fields.
   *
   * Unlike FieldValue, it doesn't directly own a reference count for immutables
   * and cowns, but instead uses the region's remembered set.
   */
  struct FieldValue
  {
  public:
    FieldValue() : tag(Value::UNINIT) {}
    ~FieldValue();

    /**
     * Get the contents of the FieldValue.
     *
     * Acquires a new reference count, owned by the return value, if necessary.
     * This never returns an ISO value. Instead iso fields are read as MUT
     * values.
     *
     * It applies viewpoint adaptation, using `parent`. Reading a mut field from
     * an IMM will return an IMM. `parent` must be one of ISO, MUT or IMM.
     */
    Value read(Value::Tag parent);

    /**
     * Exchange the contents of this FieldValue.
     *
     * `region` should be the region of the object containing this field.
     * If `value` is an immutable, it is added to the region's RememberedSet.
     */
    Value exchange(rt::Alloc* alloc, rt::Object* region, Value&& value);

    void trace(rt::ObjectStack& stack) const;

    /**
     * If this Value contains an ISO, then add it to the stack
     * Otherwise, do nothing.
     **/
    void add_isos(rt::ObjectStack& stack) const;

    friend fmt::formatter<Value>;

  private:
    Value::Tag tag;
    Value::Inner inner;
  };
}
