// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/descriptor.h"

#include "ds/helpers.h"

namespace verona::compiler
{
  using bytecode::SelectorIdx;

  class EmitProgramHeader
  {
  public:
    EmitProgramHeader(
      const Program& program,
      const Reachability& reachability,
      const SelectorTable& selectors,
      Generator& gen)
    : program(program),
      reachability(reachability),
      selectors(selectors),
      gen(gen)
    {}

    void emit_descriptor_table()
    {
      // Number of descriptors
      gen.u32(truncate<uint32_t>(reachability.entities.size()));

      size_t index = 0;
      for (const auto& [entity, info] : reachability.entities)
      {
        gen.define_relocatable(info.descriptor, index++);
        emit_descriptor(entity, info);
      }
    }

    void emit_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      switch (entity.definition->kind->value())
      {
        case Entity::Class:
        case Entity::Primitive:
          emit_class_primitive_descriptor(entity, info);
          break;

        case Entity::Interface:
          emit_interface_descriptor(entity);
          break;

          EXHAUSTIVE_SWITCH;
      }
    }

    void emit_class_primitive_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      Generator::Relocatable rel_method_slots = gen.create_relocatable();
      Generator::Relocatable rel_field_slots = gen.create_relocatable();
      Generator::Relocatable rel_field_count = gen.create_relocatable();

      gen.str(entity.instantiated_path());
      gen.u32(rel_method_slots);
      gen.u32(truncate<uint32_t>(info.methods.size()));
      gen.u32(rel_field_slots);
      gen.u32(rel_field_count);

      // Output label for finaliser for this class, if it has one.
      if (info.finaliser.label.has_value())
        gen.u32(info.finaliser.label.value());
      else
        gen.u32(0);

      uint32_t method_slots = 0;
      for (const auto& [method, info] : info.methods)
      {
        TypeList arguments;
        for (const auto& param : method.definition->signature->generics->types)
        {
          arguments.push_back(method.instantiation.types().at(param->index));
        }

        Selector selector =
          Selector::method(method.definition->name, arguments);
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

    void emit_interface_descriptor(const CodegenItem<Entity>& entity)
    {
      gen.str(entity.instantiated_path());
      gen.u32(0);
      gen.u32(0);
      gen.u32(0);
      gen.u32(0);
      gen.u32(0);
    }

    void emit_optional_special_descriptor(const std::string& name)
    {
      const EntityReachability* entity_info = nullptr;
      if (const Entity* entity = program.find_entity(name))
      {
        CodegenItem item(entity, Instantiation::empty());
        entity_info = reachability.try_find_entity(item);
      }

      if (entity_info)
        gen.u32(entity_info->descriptor);
      else
        gen.u32(bytecode::INVALID_DESCRIPTOR);
    }

    void emit_special_descriptors(const CodegenItem<Entity>& main_class)
    {
      // Index of the main descriptor
      gen.u32(reachability.find_entity(main_class).descriptor);
      // Index of the main selector
      gen.u32(selectors.get(Selector::method("main", TypeList())));

      emit_optional_special_descriptor("U64");
    }

  private:
    const Program& program;
    const Reachability& reachability;
    const SelectorTable& selectors;
    Generator& gen;
  };

  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& main)
  {
    EmitProgramHeader emit(program, reachability, selectors, gen);
    emit.emit_descriptor_table();
    emit.emit_special_descriptors(main);
  }
};
