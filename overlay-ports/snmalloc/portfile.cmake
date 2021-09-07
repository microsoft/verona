vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO microsoft/snmalloc
    REF 6c5626fe5f07b89eb6afc47d0a3abce517c8fbe1
    SHA512 39832af03e63ea29cbe1b3dda3f4542f472f63e706c32a7c85a2290876d8512d10335f01188c3cffce0ab1272d820da038d3c4392abfaa7f24957d9d4f99dc80
    HEAD_REF snmalloc2
)
vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS -DSNMALLOC_HEADER_ONLY_LIBRARY=ON
    PREFER_NINJA
)
vcpkg_install_cmake()
vcpkg_cmake_config_fixup()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/")
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/snmalloc RENAME copyright)
