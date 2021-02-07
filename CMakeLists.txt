if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
        message(FATAL_ERROR "DO NOT BUILD in-tree.")
endif()


cmake_minimum_required(VERSION 3.11)

set(PKGNAME xfoil)
set(PKGVER 6.99)
set(XFOIL_URL http://web.mit.edu/drela/Public/web/xfoil/${PKGNAME}${PKGVER}.tgz)
set(XFOIL_HASH MD5=8d6270fc4341d299a426b3ad41be9cc4)
set(XFOIL_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Xfoil)

# Download sources from MIT and AUR
include(FetchContent)
FetchContent_Declare(
        xfoil_source
        URL ${XFOIL_URL}
        URL_HASH ${XFOIL_HASH}
        SOURCE_DIR ${XFOIL_SOURCE_DIR}
)
FetchContent_Declare(
        aur_patch
        GIT_REPOSITORY https://aur.archlinux.org/xfoil.git
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/aur_patch
)
FetchContent_MakeAvailable(xfoil_source aur_patch)

# Applying patches
execute_process(
        COMMAND patch -Np1 -i ../aur_patch/xfoil-fix-write-after-end.patch
        COMMAND patch -Np1 -i ../aur_patch/xfoil-overflow.patch
        COMMAND patch -Np1 -i ../aur_patch/xfoil-build.patch
        COMMAND patch -Np1 -i ../cmake_patch/osmap_cmake.patch
        COMMAND patch -Np1 -i ../cmake_patch/cmake_lists.patch
        WORKING_DIRECTORY ${XFOIL_SOURCE_DIR}
)

project(xfoil Fortran C)

set(PKGDIR ${CMAKE_CURRENT_SOURCE_DIR}/xfoil_py/runs)
set(SHARE_DIR ${CMAKE_BINARY_DIR}/share)
set(DAT_DIR ${PKGDIR}/share/orrs/osmap.dat)
# Check if string is too big to put in osmap.f
string(LENGTH  ${PKGDIR}/share/orrs/osmap.dat PATH_SIZE)
if(PATH_SIZE GREATER 50)
        # String too long, creating new line
        string(SUBSTRING ${DAT_DIR} 0 50 OSMAP_DIR_START)
        string(SUBSTRING ${DAT_DIR} 50 -1 OSMAP_DIR_END)
        string(JOIN "\n     &" OSMAP_FINAL_DIR ${OSMAP_DIR_START} ${OSMAP_DIR_END})
else()
        set(OSMAP_FINAL_DIR ${DAT_DIR})
endif ()

# Putting osmap final dir into file
configure_file(${XFOIL_SOURCE_DIR}/orrs/src/osmap.f.in ${XFOIL_SOURCE_DIR}/orrs/src/osmap.f)

add_subdirectory(Xfoil)

# Copy binaries to xfoil_py/runs folder after install
add_custom_target(move_share_folder ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHARE_DIR}/xfoil/ ${PKGDIR}/share/
        )
add_dependencies(move_share_folder osgen)