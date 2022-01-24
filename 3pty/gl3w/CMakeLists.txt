set(src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gl3w.c
  ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/gl3w.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/glcorearb.h)

find_package (OpenGL REQUIRED)
add_library (gl3w STATIC ${src_files})
target_link_libraries (gl3w PUBLIC ${OPENGL_gl_LIBRARY})
target_include_directories (gl3w PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
