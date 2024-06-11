# =========================================================
# Static core library =====================================
# =========================================================
set(PROJECT_NAME_DIFFUSION_MODEL diffusion_model)

project(${PROJECT_NAME_DIFFUSION_MODEL} CXX)

add_library(
        ${PROJECT_NAME_DIFFUSION_MODEL}
        STATIC
        "Config/Config.cpp"
        "Diffusion/KarrasDiffusion.cpp"
        "Diffusion/Sampler.cpp"
        "Model/Model.cpp"
        "Model/Modules.cpp"
        "Trainer/Dataloader.cpp"
        "Trainer/Trainer.cpp"
        "Trainer/LRScheduler.cpp"
        "Trainer/EMA.cpp"
        "Util/FileUtil.cpp"
)

target_include_directories(
        ${PROJECT_NAME_DIFFUSION_MODEL}
        PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${EXTERNAL_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME_DIFFUSION_MODEL}
        PUBLIC
        ${PROJECT_LIBS}
)

# =========================================================
# Train executable ========================================
# =========================================================
set(PROJECT_NAME_TRAIN_EXE train)

project(${PROJECT_NAME_TRAIN_EXE} CXX)

add_executable(
        ${PROJECT_NAME_TRAIN_EXE}
        "App/Train.cpp"
)

target_include_directories(
        ${PROJECT_NAME_TRAIN_EXE}
        PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${EXTERNAL_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME_TRAIN_EXE}
        PUBLIC
        ${PROJECT_NAME_DIFFUSION_MODEL}
        ${PROJECT_LIBS}
)

# =========================================================
# Sample executable =======================================
# =========================================================
set(PROJECT_NAME_SAMPLE_EXE sample)

project(${PROJECT_NAME_SAMPLE_EXE} CXX)

add_executable(
        ${PROJECT_NAME_SAMPLE_EXE}
        "App/Sample.cpp"
)

target_include_directories(
        ${PROJECT_NAME_SAMPLE_EXE}
        PUBLIC
        ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME_SAMPLE_EXE}
        PUBLIC
        ${PROJECT_NAME_DIFFUSION_MODEL}
        ${PROJECT_LIBS}
)
