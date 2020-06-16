// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "module.h"

#include "path.h"

using namespace peg::udl;

namespace
{
  using namespace module;

  ModulePtr make_module(const std::string& name)
  {
    return std::make_shared<Module>(name);
  }

  void extract(ast::Ast& ast, std::vector<std::string>& deps)
  {
    switch (ast->tag)
    {
      case "package_loc"_:
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
    auto m = make_module(path);
    modules.emplace(canonical_path, m);
    stack.push_back(m);

    while (!stack.empty())
    {
      m = stack.back();
      stack.pop_back();
      m->ast = parser::parse(parser, m->name, ext, err);

      if (!m->ast)
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
        err << "  " <<m->name << "\n";

      err << err::end;
    }
  }

  bool run_passes(ModulePtr& m, const Passes& passes, err::Errors& err)
  {
    return dfs::post(
      m,
      [](auto& m, auto& passes, auto& err) {
        // Stop running passes when we get errors on this module, then add those
        // errors to the main error list.
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
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto m = load(parser, path, ext, err);

    if (err.empty())
      detect_cycles(m, err);

    if (err.empty())
      run_passes(m, passes, err);

    return m;
  }

  ModulePtr build(
    const std::string& grammar,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto parser = parser::create(grammar, err);

    if (!err.empty())
      return {};

    return build(parser, passes, path, ext, err);
  }
}
