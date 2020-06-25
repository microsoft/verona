// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/intern.h"
#include "compiler/source_manager.h"
#include "compiler/type.h"

#include <fstream>

namespace verona::compiler
{
  class Polarizer;
  class FreeVariablesVisitor;
  struct FreeVariables;

  class Context : public SourceManager, public TypeInterner
  {
  public:
    Context();
    ~Context();

    Polarizer& polarizer();

    const FreeVariables& free_variables(const TypePtr& type);

    template<typename... Ts>
    std::unique_ptr<std::ostream> dump(const std::string& base, Ts... args)
    {
      std::stringstream name;
      name << base;
      build_name(name, args...);
      return dump_with_name(name.str());
    }

    std::unique_ptr<std::ostream> dump_with_name(const std::string& name);

    void set_dump_path(std::string path)
    {
      dump_path_ = path;
    }

    void add_print_pattern(std::string pattern)
    {
      print_patterns_.push_back(pattern);
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

  private:
    template<typename T, typename... Ts>
    void build_name(std::stringstream& s, T head, Ts... tail)
    {
      s << "." << head;
      build_name(s, tail...);
    }
    void build_name(std::stringstream& s) {}

    bool should_print_name(std::string_view name);

    std::unique_ptr<Polarizer> polarizer_;
    std::unique_ptr<FreeVariablesVisitor> free_variables_;

    std::optional<std::string> dump_path_;
    std::vector<std::string> print_patterns_;
  };

  /**
   * RAII class that sets the thread context and restores it when this object
   * goes out of scope.
   */
  struct ThreadContext
  {
    /**
     * Constructor.  Sets the thread context to the argument value.     */
    ThreadContext(Context& c) : old_context(thread_context)
    {
      thread_context = &c;
    }

    /**
     * Returns the context object associated with the current thread.  There
     * must be a context associated with this thread when this method is
     * called.
     */
    static Context& get()
    {
      if (thread_context == nullptr)
      {
        throw std::logic_error(
          "ThreadContext::get called with no thread context");
      }
      return *thread_context;
    }

    /**
     * Deleted copy constructor and assignment.
     * These objects should not be copied or moved.
     */
    ThreadContext(const ThreadContext&) = delete;
    ThreadContext& operator=(const ThreadContext&) = delete;

    /**
     * Destructor, restores the context to the value that it held when this
     * was constructed.
     */
    ~ThreadContext()
    {
      thread_context = old_context;
    }

    /**
     * Deleted operator new - this class must be allocated on the stack.
     */
    void* operator new(size_t) = delete;

    /**
     * Deleted placement new - this class must be allocated on the stack.
     */
    void* operator new(size_t, void*) = delete;

  private:
    /**
     * The context object associated with the current thread, if one exists.
     */
    static thread_local Context* thread_context;

    /**
     * Cached version of the old context.
     */
    Context* old_context;
  };
}
