#pragma once

#include <vector>
#include "Types.hpp"

namespace sph {

struct OctreeData {
    std::vector<int32_t> tree;
    std::vector<std::vector<int32_t>> levels;
    int32_t nodeValues = 0;
};

OctreeData buildOctree(const std::vector<Vec3>& positions,
                       const std::vector<uint32_t>& alive,
                       int maxLeafSize,
                       size_t estimatedTreeSize);

} // namespace sph
