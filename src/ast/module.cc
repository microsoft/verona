// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "module.h"

#include "cli.h"
#include "path.h"

using namespace peg::udl;

namespace
{
  using namespace module;

  ModulePtr make_module(const std::string& name)
  {
    return std::make_shared<Module>(name);
  }

  // This ensures that a module has only one moduledef node, and transforms the
  // moduledef to a classdef, such that from this point on, modules are classes.
  bool moduledef(ast::Ast& ast, const std::string& path, err::Errors& err)
  {
    std::vector<ast::Ast> defs;

    for (auto& node : ast->nodes)
    {
      if (node->tag == "moduledef"_)
        defs.push_back(node);
    }

    if (defs.size() > 1)
    {
      err << defs.front() << "This module contains multiple definitions.\n";

      for (auto& def : defs)
        err << def << "Module definition here.\n";

      err << err::end;
      return false;
    }

    auto moduledef = (defs.size() > 0) ? defs.front() : ast;
    auto classdef = ast::node(moduledef, "classdef");
    auto id = ast::token(classdef, "id", "$module");
    ast::push_back(classdef, id);

    if (moduledef->tag == "moduledef"_)
    {
      ast::remove(moduledef);

      for (auto& node : moduledef->nodes)
        ast::push_back(classdef, node);

      moduledef->nodes.clear();
    }
    else
    {
      auto typeparams = ast::node(moduledef, "typeparams");
      ast::push_back(classdef, typeparams);
      auto oftype = ast::node(moduledef, "oftype");
      ast::push_back(classdef, oftype);
      auto constraints = ast::node(moduledef, "constraints");
      ast::push_back(classdef, constraints);
    }

    auto typebody = ast::node(moduledef, "typebody");
    ast::push_back(classdef, typebody);

    for (auto& node : ast->nodes)
      ast::push_back(typebody, node);

    ast->nodes.clear();
    ast = classdef;
    return true;
  }

  // This extract all module references from an Ast, and builds a vector of
  // dependency paths rooted in the Ast path.
  void extract(ast::Ast& ast, std::vector<std::string>& deps)
  {
    switch (ast->tag)
    {
      case "module_ref"_:
      {
        auto name = path::join(ast->path, ast->nodes.front()->token);

        if (path::extension(name).empty())
          name = path::to_directory(name);

        deps.push_back(name);
        return;
      }
    }

    ast::for_each(ast, extract, deps);
  }

  ModulePtr load(
    peg::parser& parser,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    std::map<std::string, ModulePtr> modules;
    std::vector<ModulePtr> stack;
    bool ok = true;

    auto canonical_path = path::canonical(path);
    auto m = make_module(path::from_platform(path));
    modules.emplace(canonical_path, m);
    stack.push_back(m);

    while (!stack.empty())
    {
      m = stack.back();
      stack.pop_back();
      m->ast = parser::parse(parser, m->name, ext, err);

      if (!m->ast || !moduledef(m->ast, m->name, err))
      {
        ok = false;
        continue;
      }

      std::vector<std::string> deps;
      extract(m->ast, deps);

      while (!deps.empty())
      {
        auto path = deps.back();
        deps.pop_back();

        canonical_path = path::canonical(path);
        auto find = modules.find(canonical_path);

        if (find != modules.end())
        {
          m->edges.push_back(find->second);
          continue;
        }

        auto dep = make_module(path);
        modules.emplace(canonical_path, dep);
        stack.push_back(dep);
        m->edges.push_back(dep);
      }
    }

    return modules.begin()->second;
  }

  void detect_cycles(ModulePtr& m, err::Errors& err)
  {
    std::vector<std::vector<ModulePtr>> cycles;

    dfs::cycles(
      m,
      [](auto& parent, auto& child, auto& cycles) {
        cycles.push_back({parent});
        auto& cycle = cycles.back();
        auto m = parent;

        while (m != child)
        {
          for (auto& m2 : m->edges)
          {
            if (m2->color == dfs::grey)
            {
              cycle.push_back(m2);
              m = m2;
              break;
            }
          }
        }

        return false;
      },
      cycles);

    for (auto& cycle : cycles)
    {
      err << "These modules cause a cyclic dependency:\n";

      for (auto& m : cycle)
        err << "  " << m->name << "\n";

      err << err::end;
    }
  }

  bool run_passes(
    ModulePtr& m,
    const std::string& stopAt,
    const pass::Passes& passes,
    err::Errors& err)
  {
    return dfs::post(
      m,
      [&stopAt](auto& m, auto& passes, auto& err) {
        // Stop running passes when we get errors on this module, then add those
        // errors to the main error list.
        if (!m->ast)
          return false;

        err::Errors lerr;
        bool ok = true;

        for (auto& pass : passes)
        {
          pass(m->ast, lerr);

          if (!lerr.empty())
          {
            err << lerr;
            ok = false;
            break;
          }

          // If asked to stop after this pass, return now.
          if (!stopAt.empty() && stopAt == pass.name)
            return ok;
        }

        return ok;
      },
      passes,
      err);
  }
}

namespace module
{
  ModulePtr build(
    peg::parser& parser,
    const std::string& stopAt,
    const pass::Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto m = load(parser, path, ext, err);

    if (err.empty())
      detect_cycles(m, err);

    // If asked to stop before any pass, return now.
    if (!stopAt.empty() && stopAt == cli::stopAtGen)
      return m;

    // Otherwise, run the passes
    if (err.empty())
      run_passes(m, stopAt, passes, err);

    return m;
  }

  ModulePtr build(
    const std::string& grammar,
    const std::string& stopAt,
    const pass::Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto parser = parser::create(grammar, err);

    if (!err.empty())
      return {};

    return build(parser, stopAt, passes, path, ext, err);
  }
}
