// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "parser.h"

#include "files.h"
#include "path.h"

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
    ast::Ast module = ast::module(path);

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto name = path::join(path, file);
      auto ast = parse(parser, name, err);

      if (ast)
      {
        for (auto& node : ast->nodes)
          ast::push_back(module, node);

        ast->nodes.clear();
      }
    }

    return module;
  }
}
