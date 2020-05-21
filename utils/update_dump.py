#!/usr/bin/env python3

import filecmp
import fnmatch
import os
import os.path
import shlex
import shutil
import subprocess
import sys
import tempfile

FILE_EXTENSION = '.verona'

class Updater:
  def __init__(self, compiler):
    self.compiler = compiler
    self.has_error = False

  def log(self, *args):
    print(*args, file=sys.stderr)

  def error(self, *args):
    self.log(*args)
    self.has_error = True

  def generate_dump(self, source, dump_dir):
    cmd = shlex.split(self.compiler) + ["--dump-path=%s" % dump_dir, source]
    self.log("Running %r" % " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
      self.error("Compiler exited with status %d" % ret)
      return False
    else:
      return True

  def update_dump_file(self, source, expected_dir, dump_dir, name):
    dump_file = os.path.join(dump_dir, name)
    target = os.path.join(expected_dir, name)

    if not os.path.exists(dump_file):
      self.error("Dump file %r was not created for %r" % (name, source))

    else:
      with open(dump_file, "rU") as dump, open(target, "rU") as targ:
        dump_data = dump.read()
        targ_data = targ.read()

      if dump_data != targ_data:
        self.log("Updating file %r" % target)
        shutil.copyfile(dump_file, target)

  def update_test(self, source):
    expected_dir, _ = os.path.splitext(source)
    dump_dir = tempfile.mkdtemp()

    try:
      if self.generate_dump(source, dump_dir):
        for name in fnmatch.filter(os.listdir(expected_dir), '*.txt'):
          self.update_dump_file(source, expected_dir, dump_dir, name)
    finally:
      shutil.rmtree(dump_dir)

  def update_dir(self, dirpath):
    for root, _, filenames in os.walk(dirpath):
       for filename in filenames:
         filepath = os.path.join(root, filename)
         stem, extension = os.path.splitext(filepath)
         if extension == FILE_EXTENSION and os.path.exists(stem):
           self.update_test(filepath)

if len(sys.argv) < 3:
  print("Usage: %s COMPILER FILES..." % sys.argv[0], file=sys.stderr)
  sys.exit(1)

updater = Updater(sys.argv[1])

for path in sys.argv[2:]:
  if os.path.isdir(path):
    updater.update_dir(path)
  else:
    updater.update_test(path)

if updater.has_error:
  sys.exit(1)
