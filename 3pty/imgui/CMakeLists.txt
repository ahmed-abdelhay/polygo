set(src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_impl_glfw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_draw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_widgets.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_demo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TextEditor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ImGuiFileBrowser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_impl_opengl3.cpp)

add_library (imgui STATIC ${src_files})
target_link_libraries (imgui PUBLIC glfw gl3w)
target_compile_definitions(imgui PUBLIC "IMGUI_IMPL_OPENGL_LOADER_GL3W")
target_include_directories (imgui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
