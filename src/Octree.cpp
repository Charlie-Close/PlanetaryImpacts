#include "sph/Octree.hpp"

#include <algorithm>
#include <cfloat>
#include <thread>
#include "sph/Parameters.hpp"

namespace sph {
namespace {

void splitGroupByAxis(const std::vector<Vec3>& positions,
                      const std::vector<int32_t>& subset,
                      int axis,
                      std::vector<int32_t>& lower,
                      std::vector<int32_t>& upper) {
    if (subset.empty()) return;
    float lo = FLT_MAX;
    float hi = -FLT_MAX;
    for (int32_t pid : subset) {
        lo = std::min(lo, positions[pid][axis]);
        hi = std::max(hi, positions[pid][axis]);
    }
    const float center = 0.5f * (lo + hi);
    size_t nLower = 0;
    size_t nUpper = 0;
    for (int32_t pid : subset) {
        if (positions[pid][axis] < center) ++nLower;
        else ++nUpper;
    }
    lower.reserve(nLower);
    upper.reserve(nUpper);
    for (int32_t pid : subset) {
        if (positions[pid][axis] < center) lower.push_back(pid);
        else upper.push_back(pid);
    }
}

int32_t buildRecursive(const std::vector<Vec3>& positions,
                       const std::vector<int32_t>& subset,
                       int level,
                       int maxLeafSize,
                       std::vector<int32_t>& tree,
                       std::vector<std::vector<int32_t>>& levels,
                       int32_t& nodeValues) {
    if (static_cast<int>(subset.size()) <= maxLeafSize || level == 100) {
        const int32_t nodeStart = static_cast<int32_t>(tree.size());
        if (level >= static_cast<int>(levels.size())) levels.resize(static_cast<size_t>(level + 1));
        levels[level].push_back(nodeStart);
        tree.push_back(static_cast<int32_t>(subset.size()));
        tree.push_back(nodeValues++);
        tree.insert(tree.end(), subset.begin(), subset.end());
        return nodeStart;
    }

    const int32_t nodeStart = static_cast<int32_t>(tree.size());
    for (int i = 0; i < 10; ++i) tree.push_back(-1);
    tree[nodeStart] = 0;
    tree[nodeStart + 1] = nodeValues++;
    if (level >= static_cast<int>(levels.size())) levels.resize(static_cast<size_t>(level + 1));
    levels[level].push_back(nodeStart);

    std::vector<int32_t> p1, p2, p11, p12, p21, p22, p111, p112, p121, p122, p211, p212, p221, p222;
    splitGroupByAxis(positions, subset, 0, p1, p2);
    if (subset.size() < 10000) {
        splitGroupByAxis(positions, p1, 1, p11, p12);
        splitGroupByAxis(positions, p2, 1, p21, p22);
        splitGroupByAxis(positions, p11, 2, p111, p112);
        splitGroupByAxis(positions, p12, 2, p121, p122);
        splitGroupByAxis(positions, p21, 2, p211, p212);
        splitGroupByAxis(positions, p22, 2, p221, p222);
    } else {
        std::thread t0(splitGroupByAxis, std::cref(positions), std::cref(p1), 1, std::ref(p11), std::ref(p12));
        std::thread t1(splitGroupByAxis, std::cref(positions), std::cref(p2), 1, std::ref(p21), std::ref(p22));
        t0.join();
        t1.join();
        std::thread t2(splitGroupByAxis, std::cref(positions), std::cref(p11), 2, std::ref(p111), std::ref(p112));
        std::thread t3(splitGroupByAxis, std::cref(positions), std::cref(p12), 2, std::ref(p121), std::ref(p122));
        std::thread t4(splitGroupByAxis, std::cref(positions), std::cref(p21), 2, std::ref(p211), std::ref(p212));
        std::thread t5(splitGroupByAxis, std::cref(positions), std::cref(p22), 2, std::ref(p221), std::ref(p222));
        t2.join();
        t3.join();
        t4.join();
        t5.join();
    }

    std::vector<int32_t> children[8] = {p111, p112, p121, p122, p211, p212, p221, p222};
    for (int c = 0; c < 8; ++c) {
        tree[nodeStart + 2 + c] = children[c].empty()
            ? -1
            : buildRecursive(positions, children[c], level + 1, maxLeafSize, tree, levels, nodeValues);
    }
    return nodeStart;
}

} // namespace

OctreeData buildOctree(const std::vector<Vec3>& positions,
                       const std::vector<uint32_t>& alive,
                       int maxLeafSize,
                       size_t estimatedTreeSize) {
    OctreeData out;
    out.tree.reserve(estimatedTreeSize);
    if (positions.empty()) return out;
    std::vector<int32_t> all;
    all.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        if (alive.empty() || alive[i] != 0u) all.push_back(static_cast<int32_t>(i));
    }
    buildRecursive(positions, all, 0, maxLeafSize, out.tree, out.levels, out.nodeValues);
    for (auto it = out.levels.begin(); it != out.levels.end();) {
        if (it->empty()) it = out.levels.erase(it);
        else ++it;
    }
    return out;
}

} // namespace sph
