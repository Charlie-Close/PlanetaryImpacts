#pragma once

#include <filesystem>
#include "Types.hpp"

namespace sph {

DataSet readHdf5(const std::filesystem::path& path);
void writeHdf5(const std::filesystem::path& path, const DataSet& data);

} // namespace sph

