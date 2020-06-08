// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "parser.h"

#include "files.h"
#include "path.h"

namespace parser
{
  std::string format_error_message(
    const std::string& path, size_t ln, size_t col, const std::string& msg)
  {
    std::stringstream ss;
    ss << path.c_str() << ":" << ln << ":" << col << ": " << msg << std::endl;
    return ss.str();
  }

  peg::parser create(const std::string& file)
  {
    return create(files::slurp(file), file);
  }

  ast::Ast parse(peg::parser& parser, const std::string& file)
  {
    return parse(parser, files::slurp(file), file);
  }

  ast::Ast
  parse(peg::parser& parser, const std::string& path, const std::string& ext)
  {
    if (!path::is_directory(path))
      return parse(parser, path);

    auto files = path::files(path);
    ast::Ast module;

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto name = path::join(path, file);
      auto ast = parse(parser, name);

      if (!module)
      {
        module = ast;
      }
      else
      {
        for (auto& node : ast->nodes)
          ast::push_back(module, node);

        ast->nodes.clear();
      }
    }

    return module;
  }
}
