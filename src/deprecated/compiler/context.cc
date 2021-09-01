// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/context.h"

#include "compiler/freevars.h"
#include "compiler/polarize.h"

namespace verona::compiler
{
  Context::Context()
  : free_variables_(std::make_unique<FreeVariablesVisitor>()),
    polarizer_(std::make_unique<Polarizer>(*this))
  {}

  Context::~Context() {}

  Polarizer& Context::polarizer()
  {
    return *polarizer_;
  }

  const FreeVariables& Context::free_variables(const TypePtr& type)
  {
    return free_variables_->free_variables(type);
  }

  bool Context::should_print_name(std::string_view name)
  {
    for (const auto& pattern : print_patterns_)
    {
      if (name.find(pattern, 0) == 0)
      {
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<std::ostream> Context::dump_with_name(const std::string& name)
  {
    if (should_print_name(name))
    {
      auto out = std::make_unique<std::ofstream>();
      out->copyfmt(std::cerr);
      out->clear(std::cerr.rdstate());
      out->std::ios::rdbuf(std::cerr.rdbuf());
      return out;
    }
    else if (dump_path_.has_value())
    {
      std::string path = *dump_path_ + "/" + name + ".txt";
      auto out = std::make_unique<std::ofstream>(path);
      if (!out->is_open())
      {
        std::cerr << "Cannot open dump file " << path << std::endl;
      }
      return out;
    }
    else
    {
      return std::make_unique<std::ofstream>();
    }
  }

  thread_local Context* ThreadContext::thread_context;
}
