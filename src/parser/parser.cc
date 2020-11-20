#include "parser.h"

#include "lexer.h"
#include "source.h"

#include "../ast/path.h"

#include <fstream>

namespace verona::parser
{
  const char* ext = "verona";

  void
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return;

    // TODO:
  }

  void parse_directory(
    const std::string& path, Node<Class>& module, err::Errors& err)
  {
    auto files = path::files(path);

    if (files.empty())
      err << "No " << ext << " files found in " << path << err::end;

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto filename = path::join(path, file);
      parse_file(filename, module, err);
    }
  }

  Node<NodeDef> parse(const std::string& path, err::Errors& err)
  {
    auto module = std::make_shared<Class>();

    if (path::is_directory(path))
    {
      parse_directory(path, module, err);
    }
    else
    {
      parse_file(path, module, err);
    }

    return module;
  }
}
