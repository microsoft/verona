
all: clang_interface clang_interface-release

clang_interface: clang_interface.cc Makefile
	clang++80 clang_interface.cc -o clang_interface \
		-fcolor-diagnostics \
		-std=c++17 \
		-fsanitize=address \
		-g \
		-I ~/llvm-project/clang/include/ \
		-I ~/llvm-project/llvm/include/ \
		-I ~/build/llvm-trunk-debug/include/ \
		-I ~/build/llvm-trunk-debug/tools/clang/include/ \
		-I /usr/local/llvm80/include \
		`~/build/llvm-trunk-debug/bin/llvm-config --cxxflags` \
		`~/build/llvm-trunk-debug/bin/llvm-config --ldflags` \
		`~/build/llvm-trunk-debug/bin/llvm-config --libs` \
		`~/build/llvm-trunk-debug/bin/llvm-config --system-libs`\
		-lclangAST \
		-lclangASTMatchers \
		-lclangAnalysis \
		-lclangBasic \
		-lclangCodeGen \
		-lclangDriver \
		-lclangEdit \
		-lclangFrontend \
		-lclangFrontendTool \
		-lclangLex \
		-lclangParse \
		-lclangRewrite \
		-lclangRewriteFrontend \
		-lclangSema \
		-lclangSerialization \
		-lclangTooling \
		-lclangCodeGen


clang_interface-release: clang_interface.cc
	clang++80 clang_interface.cc -o clang_interface-release \
		-fcolor-diagnostics \
		-std=c++17 \
		-g \
		-I ~/llvm-project/clang/include/ \
		-I ~/llvm-project/llvm/include/ \
		-I ~/build/llvm-trunk-release/include/ \
		-I ~/build/llvm-trunk-release/tools/clang/include/ \
		`~/build/llvm-trunk-release/bin/llvm-config --cxxflags` \
		`~/build/llvm-trunk-release/bin/llvm-config --ldflags` \
		`~/build/llvm-trunk-release/bin/llvm-config --libs` \
		`~/build/llvm-trunk-release/bin/llvm-config --system-libs`\
		-lclangAST \
		-lclangASTMatchers \
		-lclangAnalysis \
		-lclangBasic \
		-lclangCodeGen \
		-lclangDriver \
		-lclangEdit \
		-lclangFrontend \
		-lclangFrontendTool \
		-lclangLex \
		-lclangParse \
		-lclangRewrite \
		-lclangRewriteFrontend \
		-lclangSema \
		-lclangSerialization \
		-lclangTooling \
		-lclangCodeGen \
		-O2 -DNDEBUG


