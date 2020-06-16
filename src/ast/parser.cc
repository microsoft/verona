// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "parser.h"

#include "files.h"
#include "path.h"

using namespace peg::udl;

namespace parser
{
  peg::parser create(const std::string& file, err::Errors& err)
  {
    return create(files::slurp(file, err), file, err);
  }

  ast::Ast parse(peg::parser& parser, const std::string& file, err::Errors& err)
  {
    return parse(parser, files::slurp(file, err), file, err);
  }

  ast::Ast parse(
    peg::parser& parser,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    if (!path::is_directory(path))
      return parse(parser, path, err);

    auto files = path::files(path);
    std::vector<ast::Ast> modules;

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto name = path::join(path, file);
      auto ast = parse(parser, name, err);

      if (ast)
        modules.push_back(ast);
    }

    if (modules.empty())
      return {};

    ast::Ast moduledef = modules.front();

    for (auto& module : modules)
    {
      for (auto& node : module->nodes)
      {
        if (node->tag == "moduledef"_)
        {
          if (moduledef->tag != "moduledef"_)
          {
            moduledef = node;
          }
          else
          {
            err << node << "A module definition already exists.\n"
                << moduledef << "Previous definition is here." << err::end;
          }
        }
      }
    }

    auto classdef = ast::node(moduledef, "classdef");
    auto id = ast::token(classdef, "id", "$module:" + path);
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

    for (auto& module : modules)
    {
      for (auto& node : module->nodes)
        ast::push_back(typebody, node);

      module->nodes.clear();
    }

    return classdef;
  }
}
