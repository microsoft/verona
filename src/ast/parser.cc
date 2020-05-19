#include "parser.h"

#include "files.h"

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
}
