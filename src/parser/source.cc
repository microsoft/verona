#include "source.h"

#include "../ast/path.h"

#include <fstream>

namespace verona::parser
{
  bool Location::is(const char* text)
  {
    return this->source->contents.compare(
             this->start, this->end - this->start + 1, text) == 0;
  }

  Source load_source(const std::string& file, err::Errors& err)
  {
    std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

    if (!f)
    {
      err << "Couldn't open file " << file << err::end;
      return {};
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    auto source = std::make_shared<SourceDef>();
    source->origin = path::canonical(file);
    source->contents.resize(size);
    f.read(&source->contents[0], size);

    if (!f)
    {
      err << "Couldn't read file " << file << err::end;
      return {};
    }

    return source;
  }
}
