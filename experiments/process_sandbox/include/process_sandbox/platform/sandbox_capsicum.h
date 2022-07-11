// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#if __has_include(<sys/capsicum.h>)
#  include "../sandbox_fd_numbers.h"

#  include <array>
#  include <sys/capsicum.h>
#  include <sys/procctl.h>
namespace sandbox::platform
{
  struct SandboxCapsicum
  {
    /**
     * Apply the sandboxing policy.  This enters capability mode at which
     * point the process loses access to the global namespace and is unable
     * to open new sockets or files, except by being passed them from the
     * parent or via calls such as `openat` that take a sufficiently
     * authorized file descriptor.
     */
    static void apply_sandboxing_policy_postexec()
    {
      cap_enter();
    }

    /**
     * Restrict the rights of inherited file descriptors and start the new
     * program.  This implementation opens the libraries and provides the file
     * descriptor numbers in an environment variable so that the run-time
     * linker can open them without needing to access the global namespace.
     */
    template<size_t EnvSize, size_t LibDirSize>
    static void execve(
      const char* pathname,
      const std::array<const char*, EnvSize>& envp,
      const std::array<const char*, LibDirSize>& libdirs)
    {
      // These are passed in by environment variable, so we don't need to put
      // them in a fixed place, just after all of the others.
      // The file descriptors for the directories in libdirs
      std::array<platform::handle_t, LibDirSize> libdirfds;
      // The last file descriptor that we're going to use.  The `move_fd`
      // lambda will copy all file descriptors above this line so they can then
      // be copied into their desired location.
      int libfd = OtherLibraries;
      for (size_t i = 0; i < libdirs.size(); i++)
      {
        int fd = open(libdirs.at(i), O_DIRECTORY);
        libdirfds.at(i) = fd;
        SANDBOX_INVARIANT(
          (fd == libfd),
          "Unexpected sandbox directory fd number {}, expected {}",
          fd,
          libfd);
        libfd++;
      }
      // If we're compiling with Capsicum support, then restrict the
      // permissions on all of the file descriptors that are available to
      // untrusted code.
      auto limit_fd = [&](int fd, auto... permissions) {
        cap_rights_t rights;
        if (cap_rights_limit(fd, cap_rights_init(&rights, permissions...)) != 0)
        {
          err(1, "Failed to limit rights on file descriptor %d", fd);
        }
      };
      // Standard in is read only
      limit_fd(STDIN_FILENO, CAP_READ);
      // Standard out and error are write only.  Allow them fsync to make sure
      // that debugging messages are properly flushed.  Without this,
      // snmalloc's `message()` raises a SIGCAP and if we call it in debug
      // builds while handling the signal then we infinite loop.
      limit_fd(STDOUT_FILENO, CAP_WRITE | CAP_FSYNC);
      limit_fd(STDERR_FILENO, CAP_WRITE | CAP_FSYNC);
      // The socket is used with a call-return protocol for requesting
      // services for malloc.
      limit_fd(PageMapUpdates, CAP_WRITE, CAP_READ);
      // The shared heap can be mapped read-write, but can't be truncated.
      limit_fd(SharedMemRegion, CAP_MMAP_RW);
      limit_fd(PageMapPage, CAP_MMAP_R);
      // The library must be parseable and mappable by rtld
      limit_fd(MainLibrary, CAP_READ, CAP_FSTAT, CAP_SEEK, CAP_MMAP_RX);
      // The libraries implicitly opened from the library directories inherit
      // the permissions from the parent directory descriptors.  These need
      // the permissions required to map a library and also the permissions
      // required to search the directory to find the relevant libraries.
      for (auto libfd : libdirfds)
      {
        limit_fd(libfd, CAP_READ, CAP_FSTAT, CAP_LOOKUP, CAP_MMAP_RX);
      }
      int arg =
#  ifdef PROC_TRAPCAP_CTL_ENABLE_SIGCAP
        PROC_TRAPCAP_CTL_ENABLE_SIGCAP
#  else
        PROC_TRAPCAP_CTL_ENABLE
#  endif
        ;
      int ret = procctl(P_PID, getpid(), PROC_TRAPCAP_CTL, &arg);
      SANDBOX_INVARIANT(
        ret == 0, "Failed to register for traps on Capsicum violations");
      std::array<const char*, EnvSize + 1> env;
      std::copy(envp.begin(), envp.end(), env.begin());
      env.at(EnvSize - 1) = "LD_LIBRARY_PATH_FDS=8:9:10";
      env.at(EnvSize) = nullptr;
      static_assert(
        (LibDirSize == 3) && (OtherLibraries == 8),
        "LD_LIBRARY_PATH_FDS environment variable is incorrect");
      // The following version would allow us to enter capability mode before
      // exec, which would tighten up the security somewhat, but unfortunately
      // doesn't work due to bugs in FreeBSD's run-time linker.  It is left so
      // that it can be enabled guarded by a __FreeBSD_version check.  There
      // are two issues currently preventing this from working:
      //
      //  - The run-time linker, when invoked directly, has an image base
      //    address that places it too low in memory for it to be able to load
      //    library_runner with its default address (snmalloc makes the .bss
      //    section 256 MiB, which makes the mapping for the binary very
      //    large).  This can be worked around by setting a higher base address
      //    for `library_runner` to a large number.
      //  - In direct execution mode, the thread-local data for the main
      //    program binary is not fully set up before calling constructors in
      //    libraries.  This results in a crash in `calloc`, which tries to
      //    dereference a pointer early on.
#  if 0
      int binfd = open(pathname, O_RDONLY);
      limit_fd(binfd, CAP_READ, CAP_FSTAT, CAP_LOOKUP, CAP_MMAP_RX);
      SANDBOX_INVARIANT(
        binfd < 100,
        "Unexpected file descriptor number for {}: {}",
        pathname,
        binfd);
      char binfdstr[3];
      sprintf(binfdstr, "%d", binfd);
      const char* rtld = "/libexec/ld-elf.so.1";
      std::array<const char*, 5> argv{rtld, "-f", binfdstr, pathname, nullptr};
      int rtldfd = open(rtld, O_EXEC);
      fprintf(stderr, "Entering capability mode!\n");
      cap_enter();
      fexecve(
        rtldfd,
        const_cast<char**>(argv.data()),
        const_cast<char**>(env.data()));
      SANDBOX_INVARIANT(0, "fexecve failed: {}", strerror(errno));
#  else
      char* args[] = {const_cast<char*>(pathname), nullptr};
      ::execve(pathname, args, const_cast<char**>(env.data()));
      SANDBOX_INVARIANT(0, "Execve failed: {}", strerror(errno));
#  endif
    }
  };
}
#endif
