// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "ds/helpers.h"

#include <fmt/format.h>

/**
 * Formatter for Value objects.
 *
 * The formatter has two modes, shallow and deep. Printing a Verona object with
 * deep formatting will print all the object's fields, recursively. On the other
 * hand, with shallow formatting only the object's class name and address are
 * displayed.
 *
 * Shallow mode is used by default by leaving out any format specifier.
 * Deep mode is enabled by using '#' as a format specifier, e.g. "Object: {:#}".
 *
 * The VM's verbose loggin always uses shallow formatting, whereas the Print
 * opcode uses deep formatting.
 *
 * Note that deep formatting on a cyclic data-structure will cause infinite
 * recursion.
 */
template<>
struct fmt::formatter<verona::interpreter::Value>
{
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    auto fmt_end = std::find(ctx.begin(), ctx.end(), '}');
    if (fmt_end != ctx.begin())
    {
      char spec = *ctx.begin();
      if (spec == '#')
        recursive = true;
      else
        throw fmt::format_error("invalid format specificer");

      ctx.advance_to(std::next(ctx.begin()));
    }
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const verona::interpreter::Value& value, FormatContext& ctx)
  {
    return format_value(value.tag, value.inner, ctx.out());
  }

private:
  bool recursive = false;

  /**
   * This function accepts a "decomposed" value, allowing it to be used with
   * either a Value or a FieldValue.
   */
  template<typename OutputIt>
  OutputIt format_value(
    verona::interpreter::Value::Tag tag,
    const verona::interpreter::Value::Inner& inner,
    OutputIt it)
  {
    using namespace verona::interpreter;
    switch (tag)
    {
      case Value::UNINIT:
        return fmt::format_to(it, "uninit");

      case Value::ISO:
      case Value::MUT:
      case Value::IMM:
      {
        VMObject* object = inner.object;
        if (recursive)
          return format_object(object, it);
        else
          return fmt::format_to(
            it, "{}({})", object->descriptor()->name, object->id<false>());
      }

      case Value::COWN:
        return fmt::format_to(
          it,
          "{}({})",
          inner.cown->descriptor->name,
          inner.cown->contents->id<false>());

      case Value::COWN_UNOWNED:
        return fmt::format_to(
          it,
          "unowned-{}({})",
          inner.cown->descriptor->name,
          inner.cown->contents->id<false>());

      case Value::U64:
        return fmt::format_to(it, "{}", inner.u64);

      case Value::STRING:
        return fmt::format_to(it, "{}", inner.string());

      case Value::DESCRIPTOR:
        return fmt::format_to(it, "descriptor({})", inner.descriptor->name);

        EXHAUSTIVE_SWITCH;
    }
  }

  template<typename OutputIt>
  OutputIt format_object(verona::interpreter::VMObject* object, OutputIt it)
  {
    const verona::interpreter::VMDescriptor* descriptor = object->descriptor();
    it = fmt::format_to(it, "{}", descriptor->name);
    if (descriptor->field_count)
    {
      it = fmt::format_to(it, " {{ ");
      for (size_t i = 0; i < descriptor->field_count; i++)
      {
        if (i > 0)
          it = fmt::format_to(it, ", ");

        const verona::interpreter::FieldValue& v = object->fields[i];
        it = format_value(v.tag, v.inner, it);
      }
      it = fmt::format_to(it, " }}");
    }
    return it;
  }
};

template<>
struct fmt::formatter<verona::interpreter::Value::Tag>
{
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const verona::interpreter::Value::Tag& tag, FormatContext& ctx)
  {
    using Tag = verona::interpreter::Value::Tag;
    switch (tag)
    {
      case Tag::UNINIT:
        return fmt::format_to(ctx.out(), "UNINIT");
      case Tag::ISO:
        return fmt::format_to(ctx.out(), "ISO");
      case Tag::MUT:
        return fmt::format_to(ctx.out(), "MUT");
      case Tag::IMM:
        return fmt::format_to(ctx.out(), "IMM");
      case Tag::U64:
        return fmt::format_to(ctx.out(), "U64");
      case Tag::COWN:
        return fmt::format_to(ctx.out(), "COWN");
      case Tag::COWN_UNOWNED:
        return fmt::format_to(ctx.out(), "COWN_UNOWNED");
      case Tag::DESCRIPTOR:
        return fmt::format_to(ctx.out(), "DESCRIPTOR");
      case Tag::STRING:
        return fmt::format_to(ctx.out(), "STRING");

        EXHAUSTIVE_SWITCH
    }
  }
};
