// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "compiler/codegen/descriptor.h"

#include "ds/helpers.h"

namespace verona::compiler
{
  using bytecode::SelectorIdx;

  void emit_class_primitive_descriptor(
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& entity,
    const EntityReachability& reachability)
  {
    Generator::Relocatable rel_method_slots = gen.create_relocatable();
    Generator::Relocatable rel_field_slots = gen.create_relocatable();
    Generator::Relocatable rel_field_count = gen.create_relocatable();

    gen.str(entity.instantiated_path());
    gen.u32(rel_method_slots);
    gen.u32(truncate<uint32_t>(reachability.methods.size()));
    gen.u32(rel_field_slots);
    gen.u32(rel_field_count);
    // Output label for finaliser for this class, if it has one.
    if (reachability.finaliser.label.has_value())
      gen.u32(reachability.finaliser.label.value());
    else
      gen.u32(0);

    uint32_t method_slots = 0;
    for (const auto& [method, info] : reachability.methods)
    {
      TypeList arguments;
      for (const auto& param : method.definition->signature->generics->types)
      {
        arguments.push_back(method.instantiation.types().at(param->index));
      }

      Selector selector = Selector::method(method.definition->name, arguments);
      SelectorIdx index = selectors.get(selector);
      gen.selector(index);
      gen.u32(info.label.value());
      method_slots = std::max((uint32_t)(index + 1), method_slots);
    }

    uint32_t field_count = 0;
    uint32_t field_slots = 0;
    for (const auto& member : entity.definition->members)
    {
      if (const Field* fld = member->get_as<Field>())
      {
        SelectorIdx index = selectors.get(Selector::field(fld->name));
        gen.selector(index);
        field_slots = std::max((uint32_t)(index + 1), field_slots);
        field_count++;
      }
    }

    gen.define_relocatable(rel_method_slots, method_slots);
    gen.define_relocatable(rel_field_slots, field_slots);
    gen.define_relocatable(rel_field_count, field_count);
  }

  void
  emit_interface_descriptor(Generator& gen, const CodegenItem<Entity>& entity)
  {
    gen.str(entity.instantiated_path());
    gen.u32(0);
    gen.u32(0);
    gen.u32(0);
    gen.u32(0);
    gen.u32(0);
  }

  void emit_descriptor(
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& entity,
    const EntityReachability& reachability)
  {
    switch (entity.definition->kind->value())
    {
      case Entity::Class:
      case Entity::Primitive:
        emit_class_primitive_descriptor(selectors, gen, entity, reachability);
        break;

      case Entity::Interface:
        emit_interface_descriptor(gen, entity);
        break;

        EXHAUSTIVE_SWITCH;
    }
  }
};
