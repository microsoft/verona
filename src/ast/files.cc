#include "files.h"

namespace files
{
  std::vector<char> slurp(const std::string& file, bool optional)
  {
    std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

    if (!f)
    {
      if (optional)
      {
        return {};
      }
      else
      {
        std::cerr << "Could not open file " << file << std::endl;
        exit(-1);
      }
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<std::vector<char>::size_type>(size));
    f.read(data.data(), size);

    if (!optional && !f)
    {
      std::cerr << "Could not read file " << file << std::endl;
      exit(-1);
    }

    return data;
  }
}
