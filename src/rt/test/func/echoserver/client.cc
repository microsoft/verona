#include <iostream>
#include <verona.h>

using namespace verona::rt;

struct send_msg : public VBehaviour<send_msg>
{
  TCPSock* sock;

  send_msg(TCPSock* sock_) : sock(sock_) {}
  void f()
  {
#if 0
    auto new_sock = sock->server_accept();
    if (new_sock)
    {
      std::cout << "Received new conn" << std::endl;
      Cown::schedule<serve_b>(new_sock, new_sock);
    }

    Cown::schedule<accept_b>(sock, sock);
#endif
  }
};

struct connected : public VBehaviour<connected>
{
  TCPSock* sock;

  connected(TCPSock* sock_) : sock(sock_) {}
  void f()
  {
    int res;

    res = sock->check_connected();
    if (res == -1)
      return; // or handle the error somehow
    else if (res == 0)
      Cown::schedule<connected>(sock, sock);
    else
      Cown::schedule<send_msg>(sock, sock);
  }
};

void open_client_conn(char *ip, int port)
{
  auto client_conn = TCPSock::client_dial(ip, port);
  Cown::schedule<connected>(client_conn, client_conn);
}

int main(int argc, char** argv)
{
  uint16_t port;
  char *ip;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: ./client <ip> <port>\n");
    return -1;
  }

  ip = argv[1];
  port = atoi(argv[2]);

  auto& sched = Scheduler::get();
  Scheduler::set_detect_leaks(true);
  sched.init(1);

  sched.run_with_startup(&open_client_conn, ip, port);
}
