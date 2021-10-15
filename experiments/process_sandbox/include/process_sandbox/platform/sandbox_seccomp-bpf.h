// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "../helpers.h"

#if __has_include(<seccomp.h>)
SANDBOX_CLANG_DIAGNOSTIC_IGNORE("-Wmissing-field-initializers")
#  include <seccomp.h>
SANDBOX_CLANG_DIAGNOSTIC_POP()
#  include <assert.h>
#  include <process_sandbox/sandbox_fd_numbers.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * The seccomp-bpf implementation of the sandbox.  This splits system calls
     * into three categories:
     *
     * - Those that don't touch the global namespace and so are
     *   unconditionally allowed.
     * - Those that use a PID and so are permitted only on the child process
     *   itself.
     * - Those that are intrinsically dangerous and so are always denied.
     *   These are typically things that require root privilege.
     * - Those that might be allowed, depending on the arguments.  These are
     *   configured to trap and the signal handler then tries to perform an
     *   upcall for them.
     */
    struct SandboxSeccompBPF
    {
      /**
       * Execute the child.  This forwards to the default implementation, no
       * sandboxing is applied before the library runner starts.  The policy is
       * applied after the program starts but before it loads any untrusted
       * code.
       */
      template<size_t EnvSize, size_t LibDirSize>
      static void execve(
        const char* pathname,
        const std::array<const char*, EnvSize>& envp,
        const std::array<const char*, LibDirSize>& libdirs)
      {
        static_assert(
          PageMapPage == 4,
          "The following line must be updated if the value of PageMapPage "
          "changes");
        // Drop write permission on the shared pagemap so that the child
        // can't just mprotect the page read-write.
        int ropagemap = open("/proc/self/fd/4", O_RDONLY);
        dup2(ropagemap, PageMapPage);
        close(ropagemap);
        SandboxNoOp::execve(pathname, envp, libdirs);
      }

      /**
       * Apply the sandboxing policy.  This is done in the child after exec, to
       * avoid allowing it to be sufficiently permissive for the initial exec
       * to work.  The glibc run-time linker is not able to run with the
       * sandboxing policy enforced and so we have to start the library runner
       * and install the signal handlers that allow `open` to work before we
       * can apply the sandboxing policy.
       */
      static void apply_sandboxing_policy_postexec()
      {
        scmp_filter_ctx ctx = nullptr;
#  ifndef DEBUG_SANDBOX
        ctx = seccomp_init(SCMP_ACT_KILL);
#  else
        ctx = seccomp_init(SCMP_ACT_LOG);
#  endif
        assert(ctx != nullptr);

        bool ret = true;

        auto add_rule = [&](auto action, auto syscall, auto... rules) {
          ret &=
            (seccomp_rule_add(
               ctx, action, syscall, sizeof...(rules), rules...) == 0);
          assert(ret);
        };
        auto allow = [&](auto syscall, auto... rules) {
          add_rule(SCMP_ACT_ALLOW, syscall, rules...);
        };
        auto trap = [&](auto syscall, auto... rules) {
          add_rule(SCMP_ACT_TRAP, syscall, rules...);
        };
        // Documentation only.  Deny is the default behaviour
        auto deny = [&](auto syscall) { (void)syscall; };
        pid_t self = getpid();
        auto allow_on_self = [&](auto syscall) {
          allow(
            syscall, SCMP_A0(SCMP_CMP_EQ, static_cast<scmp_datum_t>(self), 0));
          allow(syscall, SCMP_A0(SCMP_CMP_EQ, static_cast<scmp_datum_t>(0), 0));
        };
        allow(SCMP_SYS(read));
        allow(SCMP_SYS(write));
        allow(SCMP_SYS(close));
        allow(SCMP_SYS(fstat));
        allow(SCMP_SYS(poll));
        allow(SCMP_SYS(lseek));
        allow(SCMP_SYS(mmap));
        // TODO: Check the security implications of changing permission on the
        // pagemap page
        allow(SCMP_SYS(mprotect));
        allow(SCMP_SYS(munmap));
        allow(SCMP_SYS(brk));
        allow(SCMP_SYS(rt_sigaction));
        allow(SCMP_SYS(rt_sigprocmask));
        allow(SCMP_SYS(rt_sigreturn));
        allow(SCMP_SYS(pread64));
        allow(SCMP_SYS(pwrite64));
        allow(SCMP_SYS(readv));
        allow(SCMP_SYS(writev));
        allow(SCMP_SYS(pipe));
        allow(SCMP_SYS(select));
        allow(SCMP_SYS(sched_yield));
        allow(SCMP_SYS(mremap));
        allow(SCMP_SYS(msync));
        allow(SCMP_SYS(mincore));
        // FIXME: Check security implications of these on the shared region:
        allow(SCMP_SYS(madvise));
        allow(SCMP_SYS(shmat));
        allow(SCMP_SYS(shmctl));
        allow(SCMP_SYS(pause));
        allow(SCMP_SYS(nanosleep));
        allow(SCMP_SYS(getitimer));
        allow(SCMP_SYS(alarm));
        allow(SCMP_SYS(setitimer));
        allow(SCMP_SYS(getpid));
        allow(SCMP_SYS(sendfile));
        allow(SCMP_SYS(socket));
        allow(SCMP_SYS(accept));
        allow(SCMP_SYS(sendto));
        allow(SCMP_SYS(recvfrom));
        allow(SCMP_SYS(sendmsg));
        allow(SCMP_SYS(recvmsg));
        allow(SCMP_SYS(listen));
        allow(SCMP_SYS(getsockname));
        allow(SCMP_SYS(getpeername));
        allow(SCMP_SYS(socketpair));
        // TODO: Audit this to ensure it doesn't alias bind / connect behaviour
        allow(SCMP_SYS(setsockopt));
        allow(SCMP_SYS(getsockopt));
        allow(SCMP_SYS(execve));
        allow(SCMP_SYS(exit));
        allow(SCMP_SYS(uname));
        allow(SCMP_SYS(semop));
        allow(SCMP_SYS(semctl));
        allow(SCMP_SYS(shmdt));
        allow(SCMP_SYS(msgsnd));
        allow(SCMP_SYS(msgrcv));
        allow(SCMP_SYS(msgctl));
        allow(SCMP_SYS(flock));
        allow(SCMP_SYS(fsync));
        allow(SCMP_SYS(fdatasync));
        allow(SCMP_SYS(getdents));
        allow(SCMP_SYS(getcwd));
        allow(SCMP_SYS(ftruncate));
        allow(SCMP_SYS(fchdir));
        allow(SCMP_SYS(fchmod));
        allow(SCMP_SYS(fchown));
        allow(SCMP_SYS(umask));
        allow(SCMP_SYS(gettimeofday));
        allow(SCMP_SYS(getrlimit));
        allow(SCMP_SYS(getrusage));
        allow(SCMP_SYS(sysinfo));
        allow(SCMP_SYS(times));
        allow(SCMP_SYS(getuid));
        allow(SCMP_SYS(getgid));
        allow(SCMP_SYS(setuid));
        allow(SCMP_SYS(setgid));
        allow(SCMP_SYS(geteuid));
        allow(SCMP_SYS(getegid));
        allow(SCMP_SYS(setpgid));
        allow(SCMP_SYS(getppid));
        allow(SCMP_SYS(getpgrp));
        allow(SCMP_SYS(setsid));
        allow(SCMP_SYS(setreuid));
        allow(SCMP_SYS(setregid));
        allow(SCMP_SYS(getgroups));
        allow(SCMP_SYS(setgroups));
        allow(SCMP_SYS(setresuid));
        allow(SCMP_SYS(getresuid));
        allow(SCMP_SYS(setresgid));
        allow(SCMP_SYS(getresgid));
        allow(SCMP_SYS(getpgid));
        allow(SCMP_SYS(setfsuid));
        allow(SCMP_SYS(setfsgid));
        allow(SCMP_SYS(getsid));
        allow(SCMP_SYS(capget));
        allow(SCMP_SYS(capset));
        allow(SCMP_SYS(rt_sigpending));
        allow(SCMP_SYS(rt_sigtimedwait));
        allow(SCMP_SYS(rt_sigqueueinfo));
        allow(SCMP_SYS(rt_sigsuspend));
        allow(SCMP_SYS(sigaltstack));
        allow(SCMP_SYS(getpriority));
        allow(SCMP_SYS(fstatfs));
        allow(SCMP_SYS(sched_setparam));
        allow(SCMP_SYS(sched_getparam));
        allow(SCMP_SYS(sched_setscheduler));
        allow(SCMP_SYS(sched_getscheduler));
        allow(SCMP_SYS(sched_get_priority_max));
        allow(SCMP_SYS(sched_get_priority_min));
        allow(SCMP_SYS(sched_rr_get_interval));
        allow(SCMP_SYS(mlock));
        allow(SCMP_SYS(munlock));
        allow(SCMP_SYS(mlockall));
        allow(SCMP_SYS(munlockall));
        allow(SCMP_SYS(vhangup));
        allow(SCMP_SYS(modify_ldt));
        // FIXME: x86-64 is safe, not sure about others
        allow(SCMP_SYS(arch_prctl));
        allow(SCMP_SYS(setrlimit));
        allow(SCMP_SYS(sync));
        allow(SCMP_SYS(gettid));
        allow(SCMP_SYS(readahead));
        allow(SCMP_SYS(fsetxattr));
        allow(SCMP_SYS(fgetxattr));
        allow(SCMP_SYS(flistxattr));
        allow(SCMP_SYS(fremovexattr));
        allow(SCMP_SYS(time));
        allow(SCMP_SYS(futex));
        allow(SCMP_SYS(sched_setaffinity));
        allow(SCMP_SYS(sched_getaffinity));
        allow(SCMP_SYS(set_thread_area));
        allow(SCMP_SYS(io_setup));
        allow(SCMP_SYS(io_destroy));
        allow(SCMP_SYS(io_getevents));
        allow(SCMP_SYS(io_submit));
        allow(SCMP_SYS(io_cancel));
        allow(SCMP_SYS(get_thread_area));
        allow(SCMP_SYS(epoll_create));
        allow(SCMP_SYS(epoll_ctl_old));
        allow(SCMP_SYS(epoll_wait_old));
        allow(SCMP_SYS(getdents64));
        allow(SCMP_SYS(set_tid_address));
        allow(SCMP_SYS(restart_syscall));
        allow(SCMP_SYS(semtimedop));
        allow(SCMP_SYS(fadvise64));
        allow(SCMP_SYS(timer_create));
        allow(SCMP_SYS(timer_settime));
        allow(SCMP_SYS(timer_gettime));
        allow(SCMP_SYS(timer_getoverrun));
        allow(SCMP_SYS(timer_delete));
        allow(SCMP_SYS(clock_settime));
        allow(SCMP_SYS(clock_gettime));
        allow(SCMP_SYS(clock_getres));
        allow(SCMP_SYS(clock_nanosleep));
        allow(SCMP_SYS(exit_group));
        allow(SCMP_SYS(epoll_wait));
        allow(SCMP_SYS(epoll_ctl));
        allow(SCMP_SYS(mbind));
        allow(SCMP_SYS(set_mempolicy));
        allow(SCMP_SYS(get_mempolicy));
        allow(SCMP_SYS(mq_timedsend));
        allow(SCMP_SYS(mq_timedreceive));
        allow(SCMP_SYS(mq_notify));
        allow(SCMP_SYS(mq_getsetattr));
        allow(SCMP_SYS(ioprio_set));
        allow(SCMP_SYS(ioprio_get));
        allow(SCMP_SYS(inotify_init));
        allow(SCMP_SYS(pselect6));
        allow(SCMP_SYS(ppoll));
        allow(SCMP_SYS(set_robust_list));
        allow(SCMP_SYS(get_robust_list));
        allow(SCMP_SYS(splice));
        allow(SCMP_SYS(tee));
        allow(SCMP_SYS(sync_file_range));
        allow(SCMP_SYS(vmsplice));
        allow(SCMP_SYS(utimensat));
        allow(SCMP_SYS(epoll_pwait));
        allow(SCMP_SYS(signalfd));
        allow(SCMP_SYS(timerfd_create));
        allow(SCMP_SYS(eventfd));
        allow(SCMP_SYS(fallocate));
        allow(SCMP_SYS(timerfd_settime));
        allow(SCMP_SYS(timerfd_gettime));
        allow(SCMP_SYS(accept4));
        allow(SCMP_SYS(signalfd4));
        allow(SCMP_SYS(eventfd2));
        allow(SCMP_SYS(epoll_create1));
        allow(SCMP_SYS(pipe2));
        allow(SCMP_SYS(inotify_init1));
        allow(SCMP_SYS(preadv));
        allow(SCMP_SYS(pwritev));
        allow(SCMP_SYS(rt_tgsigqueueinfo));
        allow(SCMP_SYS(recvmmsg));
        allow(SCMP_SYS(fanotify_init));
        allow(SCMP_SYS(clock_adjtime));
        allow(SCMP_SYS(syncfs));
        allow(SCMP_SYS(sendmmsg));
        allow(SCMP_SYS(setns));
        allow(SCMP_SYS(getcpu));
        allow(SCMP_SYS(sched_setattr));
        allow(SCMP_SYS(sched_getattr));
        allow(SCMP_SYS(getrandom));
        allow(SCMP_SYS(memfd_create));
        allow(SCMP_SYS(execveat));
        allow(SCMP_SYS(userfaultfd));
        allow(SCMP_SYS(membarrier));
        allow(SCMP_SYS(mlock2));
        allow(SCMP_SYS(copy_file_range));
        allow(SCMP_SYS(preadv2));
        allow(SCMP_SYS(pwritev2));
        allow(SCMP_SYS(pkey_mprotect));
        allow(SCMP_SYS(pkey_alloc));
        allow(SCMP_SYS(pkey_free));
        allow(SCMP_SYS(io_pgetevents));
        allow(SCMP_SYS(rseq));
        allow(SCMP_SYS(pidfd_send_signal));
        allow(SCMP_SYS(io_uring_setup));
        allow(SCMP_SYS(io_uring_enter));
        allow(SCMP_SYS(io_uring_register));

        // FIXME: These are probably fine, but we may want to prevent certain
        // FDs from being duplicated.
        allow(SCMP_SYS(dup));
        allow(SCMP_SYS(dup2));
        allow(SCMP_SYS(fcntl));
        allow(SCMP_SYS(dup3));

        // Some syscalls take a pid as the first argument and should be allowed
        // only on the calling process.
        allow_on_self(SCMP_SYS(prlimit64));
        allow_on_self(SCMP_SYS(move_pages));
        allow_on_self(SCMP_SYS(migrate_pages));
        allow_on_self(SCMP_SYS(tgkill));
        allow_on_self(SCMP_SYS(process_vm_readv));
        allow_on_self(SCMP_SYS(process_vm_writev));
        allow_on_self(SCMP_SYS(pidfd_open));

        // Clone is allowed to create new threads, but not to do anything else
        allow(
          SCMP_SYS(clone),
          SCMP_A0(
            SCMP_CMP_MASKED_EQ,
            static_cast<scmp_datum_t>(CLONE_THREAD),
            static_cast<scmp_datum_t>(CLONE_THREAD)));

        // All of these system calls are completely disallowed.
        deny(SCMP_SYS(fork));
        deny(SCMP_SYS(vfork));
        deny(SCMP_SYS(shutdown));
        deny(SCMP_SYS(ptrace));
        deny(SCMP_SYS(mknod));
        deny(SCMP_SYS(uselib));
        deny(SCMP_SYS(personality));
        deny(SCMP_SYS(pivot_root));
        deny(SCMP_SYS(_sysctl));
        deny(SCMP_SYS(adjtimex));
        deny(SCMP_SYS(chroot));
        deny(SCMP_SYS(acct));
        deny(SCMP_SYS(settimeofday));
        deny(SCMP_SYS(mount));
        deny(SCMP_SYS(umount2));
        deny(SCMP_SYS(swapon));
        deny(SCMP_SYS(swapoff));
        deny(SCMP_SYS(reboot));
        deny(SCMP_SYS(sethostname));
        deny(SCMP_SYS(setdomainname));
        deny(SCMP_SYS(iopl));
        deny(SCMP_SYS(ioperm));
        deny(SCMP_SYS(create_module));
        deny(SCMP_SYS(init_module));
        deny(SCMP_SYS(delete_module));
        deny(SCMP_SYS(get_kernel_syms));
        deny(SCMP_SYS(query_module));
        deny(SCMP_SYS(quotactl));
        deny(SCMP_SYS(nfsservctl));
        deny(SCMP_SYS(getpmsg));
        deny(SCMP_SYS(putpmsg));
        deny(SCMP_SYS(afs_syscall));
        deny(SCMP_SYS(tuxcall));
        deny(SCMP_SYS(security));
        deny(SCMP_SYS(tkill));
        deny(SCMP_SYS(remap_file_pages));
        deny(SCMP_SYS(lookup_dcookie));
        deny(SCMP_SYS(vserver));
        deny(SCMP_SYS(unshare));
        deny(SCMP_SYS(seccomp));
        deny(SCMP_SYS(kexec_file_load));
        deny(SCMP_SYS(bpf));
        deny(SCMP_SYS(open_tree));
        deny(SCMP_SYS(move_mount));
        deny(SCMP_SYS(fsopen));
        deny(SCMP_SYS(fsconfig));
        deny(SCMP_SYS(fsmount));
        deny(SCMP_SYS(fspick));
        deny(SCMP_SYS(kexec_load));
        // TODO: We might want to allow this if we allow the sandbox to spawn
        // children, but probably not.  These can be used to monitor any other
        // process.
        deny(SCMP_SYS(wait4));
        deny(SCMP_SYS(waitid));
        deny(SCMP_SYS(add_key));
        deny(SCMP_SYS(request_key));
        deny(SCMP_SYS(keyctl));
        // Can't check args, could potentially catch a trap and call `clone`
        // with safe args.
        deny(SCMP_SYS(clone3));

        // These system calls can be handled sometimes, depending on the
        // dynamic policy.
        trap(SCMP_SYS(open));
        trap(SCMP_SYS(stat));
        trap(SCMP_SYS(lstat));
        trap(SCMP_SYS(ioctl));
        trap(SCMP_SYS(access));
        trap(SCMP_SYS(shmget));
        trap(SCMP_SYS(connect));
        trap(SCMP_SYS(bind));
        trap(SCMP_SYS(kill));
        trap(SCMP_SYS(semget));
        trap(SCMP_SYS(msgget));
        trap(SCMP_SYS(truncate));
        trap(SCMP_SYS(chdir));
        trap(SCMP_SYS(rename));
        trap(SCMP_SYS(mkdir));
        trap(SCMP_SYS(rmdir));
        trap(SCMP_SYS(creat));
        trap(SCMP_SYS(link));
        trap(SCMP_SYS(unlink));
        trap(SCMP_SYS(symlink));
        trap(SCMP_SYS(readlink));
        trap(SCMP_SYS(chmod));
        trap(SCMP_SYS(chown));
        trap(SCMP_SYS(lchown));
        trap(SCMP_SYS(syslog));
        trap(SCMP_SYS(utime));
        trap(SCMP_SYS(ustat));
        trap(SCMP_SYS(statfs));
        trap(SCMP_SYS(sysfs));
        // FIXME: Restrict to setting the child's priority
        trap(SCMP_SYS(setpriority));
        // FIXME: There are probably some safe things you can do with this:
        trap(SCMP_SYS(prctl));
        trap(SCMP_SYS(setxattr));
        trap(SCMP_SYS(lsetxattr));
        trap(SCMP_SYS(getxattr));
        trap(SCMP_SYS(lgetxattr));
        trap(SCMP_SYS(listxattr));
        trap(SCMP_SYS(llistxattr));
        trap(SCMP_SYS(removexattr));
        trap(SCMP_SYS(lremovexattr));
        trap(SCMP_SYS(utimes));
        trap(SCMP_SYS(mq_open));
        trap(SCMP_SYS(mq_unlink));
        trap(SCMP_SYS(perf_event_open));
        trap(SCMP_SYS(fanotify_mark));
        trap(SCMP_SYS(name_to_handle_at));
        trap(SCMP_SYS(open_by_handle_at));
        trap(SCMP_SYS(renameat2));
        trap(SCMP_SYS(statx));
        trap(SCMP_SYS(openat));
        trap(SCMP_SYS(mkdirat));
        trap(SCMP_SYS(mknodat));
        trap(SCMP_SYS(fchownat));
        trap(SCMP_SYS(futimesat));
        trap(SCMP_SYS(newfstatat));
        trap(SCMP_SYS(unlinkat));
        trap(SCMP_SYS(renameat));
        trap(SCMP_SYS(linkat));
        trap(SCMP_SYS(symlinkat));
        trap(SCMP_SYS(readlinkat));
        trap(SCMP_SYS(fchmodat));
        trap(SCMP_SYS(faccessat));
        trap(SCMP_SYS(inotify_add_watch));
        trap(SCMP_SYS(inotify_rm_watch));

        assert(ctx != nullptr);
        ret &= (seccomp_load(ctx) == 0);
        seccomp_release(ctx);
        assert(ret);
      }
    };
  }
}
#endif
