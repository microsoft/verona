// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "module.h"

#include "path.h"

using namespace peg::udl;

namespace
{
  using namespace module;
  using Modules = std::map<std::string, ModulePtr>;

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
        break;
      }
    }

    ast::for_each(ast, deps, extract);
  }

  ModulePtr load(
    peg::parser& parser,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    Modules modules;
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

  bool run_passes(ModulePtr& m, const Passes& passes)
  {
    struct RunPasses : public dfs::Default<Module>
    {
      const Passes& passes;
      RunPasses(const Passes& passes) : passes(passes) {}

      bool post(ModulePtr& m)
      {
        bool ok = true;

        for (auto& pass : passes)
        {
          pass(m->ast, m->err);

          if (!m->err.empty())
          {
            ok = false;
            break;
          }
        }

        return ok;
      }
    };

    RunPasses rp(passes);
    return dfs::dfs(m, rp);
  }

  bool gather_errors(ModulePtr& m, err::Errors& err)
  {
    struct GatherErrors : public dfs::Default<Module>
    {
      err::Errors& err;
      GatherErrors(err::Errors& err) : err(err) {}

      bool post(ModulePtr& m)
      {
        err << m->err;
        return true;
      }
    };

    GatherErrors ge(err);
    return dfs::dfs(m, ge);
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
    {
      dfs::CyclicPairs<Module> pairs;
      dfs::detect_cycles(m, pairs);

      for (auto& pair : pairs)
      {
        err << "These modules cause a cyclic dependency:" << std::endl
            << "  " << pair.second->name << std::endl
            << "  " << pair.first->name << err::end;
      }
    }

    if (err.empty())
    {
      if (!run_passes(m, passes))
        gather_errors(m, err);
    }

    return m;
  }
}
