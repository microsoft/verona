#!/usr/bin/env bash

# Sets up the build dir by downloading the tar ball from Azure, unpacking on
# the external repository and changing the cmake paths to the current dir

set -e

expected_args=1
if [[ $# != $expected_args ]]; then
  echo "Usage: $0 <image-file>"
  exit 1
fi

git_root="$(git rev-parse --show-toplevel)"
if [ "$?" != "0" ]; then
  echo "Not in a git directory"
  exit 1
fi
file="$1"
image="$(basename $file .tar.gz)"

# Download build cache
if [ ! -f "$file" ]; then
  echo "Downloading $image"
  az artifacts universal download \
    --organization "https://dev.azure.com/ProjectVeronaCI/" \
    --project "22b49111-ce1d-420e-8301-3fea815478ea" \
    --scope project \
    --feed "LLVMBuild" \
    --name "$image" \
    --version "*" \
    --path /tmp
fi
if [ ! -f "$file" ]; then
  echo "$file not downloaded correctly"
  exit 1
fi

# Unpack into llvm's directory
llvm_root="$git_root/external/llvm-project"
echo "Rebuilding LLVM's build directory: $llvm_root/build"
rm -rf "$llvm_root/build"
tar zxf "$file" --directory "$llvm_root"

# Find what's the LLVM's build root dir
# cmake_install.cmake is quite helpful, its first line is:
# Install script for directory: C:/agent/_work/1/s/llvm
devops_root="$(head -n 1 $llvm_root/build/cmake_install.cmake | perl -ne 'print $1 if / ([A-Z]?:?\/.*?)\/llvm/')"

# Change LLVMConfig with current path
echo "Replacing CMake paths: $devops_root -> $llvm_root"
for file in $(find "$llvm_root/build" -name \*.cmake); do
  perl -pi -e "s,$devops_root,$llvm_root,g" "$file"
done
