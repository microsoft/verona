// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef _MSC_VER
#  include <DbgHelp.h>
#  include <windows.h>
#  pragma comment(lib, "dbghelp.lib")
#elif defined(USE_EXECINFO)
#  include "threadping.h"

#  include <condition_variable>
#  include <csignal>
#  include <cxxabi.h>
#  include <execinfo.h>
#  include <thread>
#endif

#include "../ds/asymlock.h"
#include "ds/morebits.h"

#include <iomanip>
#include <iostream>
#include <snmalloc/snmalloc.h>
#include <sstream>

namespace Logging
{
#ifdef USE_SYSTEMATIC_TESTING
  static constexpr bool systematic = true;
#else
  static constexpr bool systematic = false;
#endif

#ifdef USE_FLIGHT_RECORDER
  static constexpr bool flight_recorder = true;
#else
  static constexpr bool flight_recorder = false;
#endif

  struct Header
  {
    size_t time;
    size_t items;
  };

  struct Item
  {
    size_t value;
    std::ostream& (*pp)(std::ostream&, size_t const&);
  };

  union Entry
  {
    Header header;
    Item item;
  };

  // Filled in later by the scheduler thread
  std::string get_systematic_id();

  class LocalLog : public snmalloc::Pooled<LocalLog>
  {
  private:
    friend class ThreadLocalLog;
    friend class SysLog;

#ifdef USE_FLIGHT_RECORDER
    static constexpr size_t size =
      (1 << 18) - 3 - (sizeof(verona::rt::AsymmetricLock) / 8);
#else
    static constexpr size_t size = 1;
#endif
    size_t index;
    size_t working_index;
    std::string systematic_id = "";
    verona::rt::AsymmetricLock alock;

    Entry log[size];

  public:
    LocalLog()
    {
      if constexpr (flight_recorder)
      {
        reset();
      }
    }

  private:
    static size_t get_start()
    {
      static size_t start = snmalloc::Aal::tick();
      return start;
    }

    void add(size_t pp, size_t val)
    {
      alock.internal_acquire();
      working_index = verona::rt::bits::inc_mod(working_index, size);
      log[working_index].header.time = val;
      log[working_index].header.items = pp;
      alock.internal_release();
    }

    void eject()
    {
      alock.internal_acquire();
      systematic_id = get_systematic_id();
      working_index = verona::rt::bits::inc_mod(working_index, size);
      log[working_index].header.time = snmalloc::Aal::tick() - get_start();
      log[working_index].header.items = (working_index - index + size) % size;
      index = working_index;
      alock.internal_release();
    }

    void suspend_logging(bool external)
    {
      if (external)
        alock.external_acquire();

      working_index = size - 1;
    }

    void resume_logging(bool external)
    {
      if (external)
        alock.external_release();

      reset();
    }

    void reset()
    {
      log[0].header.time = 0;
      log[0].header.items = 0;
      index = 0;
      working_index = 0;
    }

    bool peek_time(size_t& time)
    {
      auto entry_size = log[index % size].header.items;

      if (entry_size == 0)
        return false;

      if (working_index <= entry_size)
        return false;

      time = log[index % size].header.time;
      return true;
    }

    void pop_and_print(std::ostream& o)
    {
      size_t time;
      assert(peek_time(time));

      size_t entry_size = log[index % size].header.items;

      time = log[index % size].header.time;

      o << systematic_id;

      index = (index - entry_size + size) % size;
      working_index = working_index - entry_size;

      for (size_t n = 1; n < entry_size; n++)
      {
        auto pp = log[(index + n) % size].item.pp;
        size_t value = log[(index + n) % size].item.value;
        (*pp)(o, value);
      }

      o << " (" << time << ")" << std::endl;
    }
  };

  using LocalLogPool = snmalloc::Pool<LocalLog, snmalloc::Alloc::StateHandle>;

  class ThreadLocalLog
  {
  private:
    friend class SysLog;

    LocalLog* log = nullptr;
#ifdef USE_FLIGHT_RECORDER
    ThreadLocalLog() : log(LocalLogPool::acquire()) {}

    ~ThreadLocalLog()
    {
      LocalLogPool::release(log);
    }
#endif

  public:
    static ThreadLocalLog& get()
    {
      static thread_local ThreadLocalLog mine;
      return mine;
    }

    static void dump(std::ostream& o)
    {
      if constexpr (flight_recorder)
      {
        o << "Crash log begins with most recent events" << std::endl;

        o << "THIS IS BACKWARDS COMPARED TO THE NORMAL LOG!" << std::endl;

        // Set up all logs for dumping
        auto curr = LocalLogPool::iterate();
        auto mine = get().log;

        while (curr != nullptr)
        {
          curr->suspend_logging(curr != mine);
          curr = LocalLogPool::iterate(curr);
        }

        LocalLog* next = nullptr;
        while (true)
        {
          next = nullptr;
          size_t t1 = 0;
          curr = LocalLogPool::iterate();

          while (curr != nullptr)
          {
            size_t t2;
            if (curr->peek_time(t2))
            {
              if (next == nullptr || t1 < t2)
              {
                next = curr;
                t1 = t2;
              }
            }
            curr = LocalLogPool::iterate(curr);
          }

          if (next == nullptr)
            break;

          next->pop_and_print(o);
        }

        curr = LocalLogPool::iterate();
        while (curr != nullptr)
        {
          curr->resume_logging(curr != mine);
          curr = LocalLogPool::iterate(curr);
        }

        o.flush();
      }
    }
  };

  template<typename T>
  static std::ostream& pretty_printer(std::ostream& os, T const& e)
  {
    return os << e;
  }
  class SysLog
  {
  private:
    std::ostream* o;
    bool first;

    static std::stringstream& get_ss()
    {
      static thread_local std::stringstream ss;
      return ss;
    }

    inline static bool& get_logging()
    {
      static bool logging = false;
      return logging;
    }

    template<typename T>
    inline SysLog& inner_cons(const T& value)
    {
      static_assert(sizeof(T) <= sizeof(size_t));

      if constexpr (systematic)
      {
        if (get_logging())
        {
          if (first)
          {
            get_ss() << get_systematic_id();
            first = false;
          }

          get_ss() << value;
        }
      }

      if constexpr (flight_recorder)
      {
        std::ostream& (*pp)(std::ostream & os, T const& e) = &(pretty_printer);

        size_t flat_value = (size_t)value;

        LocalLog* log = ThreadLocalLog::get().log;
        log->add((size_t)pp, (size_t)flat_value);
      }

      return *this;
    }

  public:
#ifdef USE_SYSTEMATIC_TESTING
    SysLog()
    {
      if constexpr (systematic)
      {
        o = &std::cout;
        first = true;
      }
    }
#endif

    static void dump_flight_recorder(std::string id = "")
    {
      static snmalloc::FlagWord dump_in_progress;

      snmalloc::FlagLock f(dump_in_progress);

      if constexpr (flight_recorder)
      {
        std::cerr << "Dump started by " << (id != "" ? id : get_systematic_id())
                  << std::endl;
        ThreadLocalLog::dump(std::cerr);
        std::cerr << "Dump complete!" << std::endl;
      }
    }

    inline SysLog& operator<<(const char* value)
    {
      return inner_cons(value);
    }

    inline SysLog& operator<<(const void* value)
    {
      return inner_cons(value);
    }

    template<typename T>
    inline SysLog& operator<<(const T& value)
    {
      static_assert(sizeof(T) <= sizeof(size_t));
      return inner_cons(value);
    }

    inline SysLog& operator<<(std::ostream& (*f)(std::ostream&))
    {
      if constexpr (systematic)
      {
        if (get_logging())
        {
          get_ss() << f;
          *o << get_ss().str();
          get_ss().str(""); // Clear the stream
          o->flush();
          first = true;
        }
      }
      if constexpr (flight_recorder)
      {
        ThreadLocalLog::get().log->eject();
      }
      return *this;
    }

    inline static void enable_logging()
    {
      get_logging() = true;
    }
  };

#if defined(CI_BUILD) && defined(_MSC_VER)
  inline LONG ExceptionHandler(_EXCEPTION_POINTERS* ExceptionInfo)
  {
    // On any exception dump the flight recorder
    // TODO:  Out of memory filtering
    // TODO:  Handle crashes in the dump method.
    // TODO:  Hold other threads here until this one is finished.
    // TODO:  Possibly add stack tracing to the dump
    // TODO:  Cross platform
    // TODO:  Check it is a runtime thread
    (void)ExceptionInfo;

    DWORD error;
    HANDLE hProcess = GetCurrentProcess();

    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;

    pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    pSymbol->MaxNameLen = MAX_SYM_NAME;

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);

    if (!SymInitialize(hProcess, NULL, TRUE))
    {
      // SymInitialize failed
      error = GetLastError();
      printf("SymInitialize returned error : %d\n", error);
      return FALSE;
    }

    void* stack[1024];
    DWORD count = CaptureStackBackTrace(0, 1024, stack, NULL);
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

    for (int i = 0; count > 0; count--, i++)
    {
      DWORD64 dwDisplacement = 0;
      DWORD64 dwAddress = (DWORD64)stack[i];

      if (SymFromAddr(hProcess, dwAddress, &dwDisplacement, pSymbol))
      {
        DWORD dwDisplacement2 = 0;
        if (SymGetLineFromAddr64(hProcess, dwAddress, &dwDisplacement2, &line))
        {
          std::cerr << "Frame: " << pSymbol->Name << " (" << line.FileName
                    << ": " << line.LineNumber << ")" << std::endl;
        }
        else
        {
          std::cerr << "Frame: " << pSymbol->Name << std::endl;
        }
      }
      else
      {
        error = GetLastError();
        std::cerr << "SymFromAddr returned error : " << error << std::endl;
      }
    }

    SysLog::dump_flight_recorder();

    return EXCEPTION_CONTINUE_SEARCH;
  }

#elif defined(CI_BUILD) && defined(USE_EXECINFO)
  static void* stack_frames = nullptr;
  static int n_frames = 0;
  static std::string systematic_id = "";

  inline static void crash_dump()
  {
    // This can't happen as this is only in response to a ping,
    // but GCC complains, and this is not fast path, so additional
    // check is not a problem.
    if (stack_frames == nullptr)
      abort();

    // Stop handling abort signals.
    auto* sa = new struct sigaction;
    sa->sa_handler = SIG_DFL;
    sigaction(SIGABRT, sa, nullptr);
    sigaction(SIGINT, sa, nullptr);

    // Attempt to print stack trace
    auto syms = backtrace_symbols((void* const*)stack_frames, n_frames);
    if (syms != nullptr)
    {
      constexpr size_t buf_size = 1024;
      char buf[buf_size];
      auto demangle_buf = static_cast<char*>(malloc(sizeof(char) * buf_size));
      for (auto i = 2; i < n_frames; i++)
      {
        auto* sym = syms[i];
#  ifdef __APPLE__
        // macOS symbol format: index  module   address function + offset
        auto* mangled_end = strrchr(sym, '+') - 1;
        *mangled_end = 0;
        auto* mangled_begin = strrchr(sym, ' ') + 1;
        *mangled_end = ' ';
#  else
        // symbol format: module(function+offset) [address]
        auto* mangled_begin = strchr(sym, '(') + 1;
        auto* mangled_end = strchr(sym, '+');
#  endif
        auto* sym_end = sym + strlen(sym);
        if (
          (mangled_begin < sym) || (mangled_end > sym_end) ||
          (mangled_end <= mangled_begin))
        {
          std::cerr << sym << std::endl;
          continue;
        }
        size_t mangled_len = (size_t)(mangled_end - mangled_begin);
        strncpy(buf, mangled_begin, mangled_len);
        buf[mangled_len] = 0;
        auto err = 0;
        auto size = buf_size;
        char* demangled = abi::__cxa_demangle(buf, demangle_buf, &size, &err);
        if (!err)
        {
          std::cerr.write(sym, mangled_begin - sym);
          std::cerr << demangled << mangled_end << std::endl;
        }
        else
        {
          std::cerr << sym << std::endl;
        }
      }
    }
    SysLog::dump_flight_recorder(systematic_id);
    abort();
  }

  /// Encapsulates thread that handles crash dump
  static verona::rt::ThreadPing crash_thread{&crash_dump};

  inline static void signal_handler(int sig, siginfo_t*, void*)
  {
    static std::atomic_flag run_already{};

    constexpr size_t max_stack_frames = 64;
    void* frames[max_stack_frames];

    // Ignore subsequent calls.
    if (!run_already.test_and_set())
    {
      auto str = strsignal(sig);

      // We're ignoring the result of write, as there's not much we can do if it
      // fails. We're about to crash anyway.
      auto s1 = write(1, str, strlen(str));
      auto s2 = write(1, "\n", 1);
      snmalloc::UNUSED(s1 + s2);

      // Set up data for the crash dump
      n_frames = backtrace(frames, max_stack_frames);
      stack_frames = frames;
      systematic_id = get_systematic_id();

      // Nudge crash dump thread to output data.
      crash_thread.ping();

      // Need to not return so frames[max_stack_frames] still exists.
      while (true)
      {
        sleep(1000);
      }
    }
  }
#endif

  inline static void enable_crash_logging()
  {
#if defined(CI_BUILD) && defined(_MSC_VER)
    AddVectoredExceptionHandler(0, &ExceptionHandler);
#elif defined(CI_BUILD) && defined(USE_EXECINFO)
    static struct sigaction sa;
    sa.sa_sigaction = signal_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGSYS, &sa, nullptr);
#endif
  }

  inline static void enable_logging()
  {
    SysLog::enable_logging();
  }

  inline SysLog& cout()
  {
    static SysLog cout_log;
    return cout_log;
  }

  inline std::ostream& endl(std::ostream& os)
  {
    if constexpr (systematic || flight_recorder)
    {
      os << std::endl;
    }

    return os;
  }
} // namespace Logging
