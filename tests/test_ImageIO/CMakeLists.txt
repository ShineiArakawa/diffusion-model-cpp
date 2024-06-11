project(test_ImageIO CXX)

add_executable(
        ${PROJECT_NAME}
        "main.cpp"
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${EXTERNAL_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC
        diffusion_model
        ${PROJECT_LIBS}
)