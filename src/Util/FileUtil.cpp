#include <DiffusionModelC++/Util/FileUtil.hpp>

namespace dmcpp {
namespace util {

std::string FileUtil::join(const std::string basePath, const std::string additional) {
  Path_t path_basePath = generic_fs::absolute(Path_t(basePath));
  return (path_basePath / Path_t(additional)).string();
}

std::string FileUtil::absPath(const std::string path) { return generic_fs::absolute(Path_t(path)).string(); }

std::string FileUtil::dirPath(const std::string path) { return generic_fs::absolute(Path_t(path)).parent_path().string(); }

std::string FileUtil::baseName(const std::string path) { return generic_fs::absolute(Path_t(path)).filename().string(); }

std::string FileUtil::extension(const std::string path) { return Path_t(path).extension().string(); }

std::string FileUtil::cwd() { return generic_fs::current_path().string(); }

void FileUtil::mkdirs(const std::string path) {
  Path_t path_basePath = generic_fs::absolute(Path_t(path));
  generic_fs::create_directories(path_basePath);
}

bool FileUtil::exists(const std::string path) { return generic_fs::exists(generic_fs::absolute(Path_t(path))); }

bool FileUtil::isFile(const std::string path) { return generic_fs::is_regular_file(generic_fs::absolute(Path_t(path))); }

bool FileUtil::isAbsolute(const std::string path) { return Path_t(path).is_absolute(); }

std::string FileUtil::getTimeStamp() {
  const time_t t = time(NULL);
  const tm *local = localtime(&t);

  char buf[128];
  strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", local);

  std::string timeStamp(buf);

  return timeStamp;
}

}  // namespace util
}  // namespace dmcpp