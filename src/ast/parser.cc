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
    ast::Ast result;

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto name = path::join(path, file);
      auto ast = parse(parser, name, err);

      if (result)
      {
        for (auto& node : ast->nodes)
          ast::push_back(result, node);

        ast->nodes.clear();
      }
      else
      {
        result = ast;
      }
    }

    return result;
  }
}
