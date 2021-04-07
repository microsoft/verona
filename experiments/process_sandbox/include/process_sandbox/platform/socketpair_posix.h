// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef __unix__
#  include <sys/socket.h>
#  include <sys/types.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * Type for creating pairs of connected sockets.
     */
    class SocketPairPosix
    {
    public:
      /**
       * The wrapper around a single socket.  This inherits from `Handle` and so
       * all lifetime management works automatically.
       */
      class Socket : public Handle
      {
        /**
         * Send a message containing `data_len` bytes of data starting at
         * `data`.  Optionally also send a file descriptor.
         */
        bool send_msg(const void* data, size_t data_len, int send_fd = -1)
        {
          struct msghdr header;
          struct iovec iov
          {
            const_cast<void*>(data), data_len
          };
          char ctrl_buf[CMSG_SPACE(sizeof(int))] = {0};
          // Point-to-point socket, no destination address
          header.msg_name = nullptr;
          header.msg_namelen = 0;
          header.msg_flags = 0;
          header.msg_iov = &iov;
          header.msg_iovlen = 1;
          if (send_fd >= 0)
          {
            header.msg_controllen = sizeof(ctrl_buf);
            header.msg_control = ctrl_buf;

            struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
            cmsg->cmsg_level = SOL_SOCKET;
            cmsg->cmsg_type = SCM_RIGHTS;
            cmsg->cmsg_len = CMSG_LEN(sizeof(int));
            int* fd_ptr = reinterpret_cast<int*>(CMSG_DATA(cmsg));
            *fd_ptr = send_fd;
          }
          else
          {
            header.msg_controllen = 0;
            header.msg_control = nullptr;
          }
          int ret;
          do
          {
            ret = sendmsg(fd, &header, MSG_NOSIGNAL);
          } while ((ret == -1) && (errno == EAGAIN));
          return ret != -1;
        }

        /**
         * Receive a message, `data_len` bytes long into `data`.  If `recv_fd`
         * is not a null pointer, try to receive a file descriptor there.  The
         * `int` pointed to by this parameter is set to -1 if no file
         * descriptor is received.
         */
        bool receive_msg(void* data, size_t data_len, int* recv_fd = nullptr)
        {
          struct msghdr header;
          struct iovec iov
          {
            const_cast<void*>(data), data_len
          };
          char ctrl_buf[CMSG_SPACE(sizeof(int))] = {0};
          // Point-to-point socket, no source address
          header.msg_name = nullptr;
          header.msg_namelen = 0;
          header.msg_flags = 0;
          header.msg_iov = &iov;
          header.msg_iovlen = 1;
          if (recv_fd != nullptr)
          {
            header.msg_controllen = sizeof(ctrl_buf);
            header.msg_control = ctrl_buf;
            *recv_fd = -1;
          }
          else
          {
            header.msg_controllen = 0;
            header.msg_control = nullptr;
          }
          int ret;
          do
          {
            ret = recvmsg(fd, &header, MSG_NOSIGNAL);
          } while ((ret == -1) && (errno == EAGAIN));
          if (
            (ret != -1) && (recv_fd != nullptr) && (header.msg_controllen > 0))
          {
            struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
            if (
              (cmsg->cmsg_level == SOL_SOCKET) &&
              (cmsg->cmsg_type == SCM_RIGHTS) &&
              (cmsg->cmsg_len == CMSG_LEN(sizeof(int))))
            {
              int* fd_ptr = reinterpret_cast<int*>(CMSG_DATA(cmsg));
              *recv_fd = *fd_ptr;
            }
          }
          return ret != -1;
        }

      public:
        /**
         * Export all constructors from `Handle`.
         */
        using Handle::Handle;

        /**
         * Send a message containing `data_len` bytes of data starting at
         * `data`.
         */
        bool send(const void* data, size_t data_len)
        {
          return send_msg(data, data_len);
        }

        /**
         * Send a message containing `data_len` bytes of data starting at
         * `data`, accompanied by a copy of the file descriptor owned by `h`.
         */
        bool send(const void* data, size_t data_len, const Handle& h)
        {
          return send_msg(data, data_len, h.fd);
        }

        /**
         * Receive a message, `data_len` bytes long into `data`.
         */
        bool receive(void* data, size_t data_len)
        {
          return receive_msg(data, data_len);
        }

        /**
         * Receive a message, `data_len` bytes long into `data`.  If the
         * message is accompanied by a file descriptor, store it in `h`,
         * otherwise reset `h` to an invalid file descriptor.  The caller can
         * use `h.is_valid()` to determine whether a file descriptor was
         * received.
         */
        bool receive(void* data, size_t data_len, Handle& h)
        {
          int recv_fd;
          bool ret = receive_msg(data, data_len, &recv_fd);
          if (ret)
          {
            h.reset(recv_fd);
          }
          return ret;
        }
      };

      /**
       * Create a pair of connected sockets.
       */
      static std::pair<Socket, Socket> create()
      {
        int socks[2];
        if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, socks))
        {
          err(1, "Failed to create socket pair");
        }
        return {Socket(socks[0]), Socket(socks[1])};
      }
    };
  }
}
#endif
