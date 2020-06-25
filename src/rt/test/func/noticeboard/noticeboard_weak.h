// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
namespace noticeboard_weak
{
  struct C : public V<C>
  {
  public:
    int x = 0;

    C(int x_) : x(x_) {}
  };

  struct Writer : public VCown<Writer>
  {
  public:
    Noticeboard<Object*> box_0;
    Noticeboard<Object*> box_1;

    Writer(Object* c_0, Object* c_1) : box_0{c_0}, box_1{c_1}
    {
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      register_noticeboard(&box_0);
      register_noticeboard(&box_1);
#endif
    }

    void trace(ObjectStack& fields) const
    {
      box_0.trace(fields);
      box_1.trace(fields);
    }
  };

  struct WriterLoop : public VAction<WriterLoop>
  {
    Writer* writer;
    WriterLoop(Writer* writer) : writer(writer) {}

    void f()
    {
      auto* alloc = ThreadAlloc::get();

      auto c_0 = new C(1);
      Freeze::apply(alloc, c_0);
      auto c_1 = new C(2);
      Freeze::apply(alloc, c_1);

      writer->box_0.update(alloc, c_0);
      writer->box_1.update(alloc, c_1);
    }
  };

  struct Reader : public VCown<Reader>
  {
  public:
    Noticeboard<Object*>* box_0;
    Noticeboard<Object*>* box_1;

    Reader(Noticeboard<Object*>* box_0_, Noticeboard<Object*>* box_1_)
    : box_0{box_0_}, box_1{box_1_}
    {}
  };

  Reader* g_reader = nullptr;
  Writer* g_writer = nullptr;

  struct ReaderLoop : public VAction<ReaderLoop>
  {
    Reader* reader;
    ReaderLoop(Reader* reader) : reader(reader) {}

    void f()
    {
      auto* alloc = ThreadAlloc::get();

      auto c_1 = (C*)reader->box_1->peek(alloc);
      auto c_0 = (C*)reader->box_0->peek(alloc);

      // expected assertion failure; write to c_1 was picked up before c_0
      // if (c_1->x == 2) {
      //   assert(c_0->x == 1);
      // }

      // out of scope
      Immutable::release(alloc, c_0);
      Immutable::release(alloc, c_1);

      Cown::release(alloc, g_reader);
      Cown::release(alloc, g_writer);
    }
  };

  void run_test()
  {
    auto* alloc = ThreadAlloc::get();

    auto c_0 = new (alloc) C(0);
    auto c_1 = new (alloc) C(1);
    Freeze::apply(alloc, c_0);
    Freeze::apply(alloc, c_1);

    g_writer = new Writer(c_0, c_1);
    g_reader = new Reader(&g_writer->box_0, &g_writer->box_1);

    Cown::schedule<ReaderLoop>(g_reader, g_reader);
    Cown::schedule<WriterLoop>(g_writer, g_writer);
  }
}
