#pragma once

#include <filesystem>
#include <string>

namespace dmcpp {
namespace util {

namespace generic_fs = std::filesystem;

class FileUtil {
  using Path_t = std::filesystem::path;

 public:
  static std::string join(const std::string, const std::string);
  static std::string dirPath(const std::string);
  static std::string baseName(const std::string);
  static std::string extension(const std::string);
  static std::string absPath(const std::string);
  static std::string cwd();
  static void mkdirs(const std::string);
  static bool exists(const std::string);
  static bool isFile(const std::string);
  static bool isAbsolute(const std::string);
  static std::string getTimeStamp();
};

}  // namespace util
}  // namespace dmcpp