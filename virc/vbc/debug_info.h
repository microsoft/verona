#pragma once

#include "../model/name_table.h"
#include "encoder.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <vir.h>

namespace virc::vbc_backend
{
  class DebugInfo
  {
  private:
    ByteBuffer& code;
    const std::vector<std::filesystem::path>& source_paths;
    ByteBuffer data;
    std::map<ST::Index, trieste::Source> sources;
    trieste::Source current_source;
    size_t source_offset = 0;
    size_t last_pc = 0;
    bool explicit_location = false;

    void advance();

  public:
    DebugInfo(
      ByteBuffer& code, const std::vector<std::filesystem::path>& source_paths)
    : code(code), source_paths(source_paths)
    {}

    size_t size() const;
    ByteBuffer& output();
    void begin_function();
    void record_explicit_file(trieste::Node file);
    void record_explicit_offset(size_t offset);
    void record_statement(trieste::Node statement);
    void finish_function(trieste::Node terminator);
    void
    write_to(std::ofstream& output, const std::filesystem::path& output_path);
  };
}