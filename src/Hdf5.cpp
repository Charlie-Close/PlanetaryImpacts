#include "sph/Hdf5.hpp"

#include <hdf5.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sph {
namespace {

struct H5Handle {
    hid_t id = -1;
    herr_t (*close)(hid_t) = nullptr;
    H5Handle() = default;
    H5Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) {}
    ~H5Handle() { if (id >= 0 && close) close(id); }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept : id(other.id), close(other.close) { other.id = -1; }
};

bool datasetExists(hid_t group, const char* name) {
    return H5Lexists(group, name, H5P_DEFAULT) > 0;
}

std::vector<Vec3> readVec3(hid_t group, const char* name) {
    H5Handle dataset(H5Dopen(group, name, H5P_DEFAULT), H5Dclose);
    if (dataset.id < 0) throw std::runtime_error(std::string("Could not open dataset ") + name);
    H5Handle space(H5Dget_space(dataset.id), H5Sclose);
    hsize_t dims[2] = {};
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    if (dims[1] != 3) throw std::runtime_error(std::string("Dataset is not Nx3: ") + name);

    std::vector<float> flat(dims[0] * 3);
    if (H5Dread(dataset.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()) < 0) {
        throw std::runtime_error(std::string("Could not read dataset ") + name);
    }

    std::vector<Vec3> out(dims[0]);
    for (hsize_t i = 0; i < dims[0]; ++i) {
        out[i] = makeVec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
    }
    return out;
}

std::vector<float> readFloat(hid_t group, const char* name) {
    H5Handle dataset(H5Dopen(group, name, H5P_DEFAULT), H5Dclose);
    if (dataset.id < 0) throw std::runtime_error(std::string("Could not open dataset ") + name);
    H5Handle space(H5Dget_space(dataset.id), H5Sclose);
    hsize_t dims[1] = {};
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    std::vector<float> out(dims[0]);
    if (H5Dread(dataset.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        throw std::runtime_error(std::string("Could not read dataset ") + name);
    }
    return out;
}

std::vector<int32_t> readInt32(hid_t group, const char* name) {
    H5Handle dataset(H5Dopen(group, name, H5P_DEFAULT), H5Dclose);
    if (dataset.id < 0) throw std::runtime_error(std::string("Could not open dataset ") + name);
    H5Handle space(H5Dget_space(dataset.id), H5Sclose);
    hsize_t dims[1] = {};
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    std::vector<int32_t> out(dims[0]);
    if (H5Dread(dataset.id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        throw std::runtime_error(std::string("Could not read dataset ") + name);
    }
    return out;
}

void writeVec3(hid_t group, const char* name, const std::vector<Vec3>& data) {
    hsize_t dims[2] = {data.size(), 3};
    H5Handle space(H5Screate_simple(2, dims, nullptr), H5Sclose);
    H5Handle dataset(H5Dcreate(group, name, H5T_NATIVE_FLOAT, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    if (dataset.id < 0) throw std::runtime_error(std::string("Could not create dataset ") + name);
    std::vector<float> flat(data.size() * 3);
    for (size_t i = 0; i < data.size(); ++i) {
        flat[i * 3] = data[i].x;
        flat[i * 3 + 1] = data[i].y;
        flat[i * 3 + 2] = data[i].z;
    }
    H5Dwrite(dataset.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
}

template <typename T>
void write1d(hid_t group, const char* name, hid_t h5Type, const std::vector<T>& data) {
    hsize_t dims[1] = {data.size()};
    H5Handle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
    H5Handle dataset(H5Dcreate(group, name, h5Type, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    if (dataset.id < 0) throw std::runtime_error(std::string("Could not create dataset ") + name);
    H5Dwrite(dataset.id, h5Type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
}

} // namespace

DataSet readHdf5(const std::filesystem::path& path) {
    H5Handle file(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (file.id < 0) throw std::runtime_error("Could not open HDF5 file: " + path.string());
    H5Handle group(H5Gopen(file.id, "PartType0", H5P_DEFAULT), H5Gclose);
    if (group.id < 0) throw std::runtime_error("Could not open PartType0 in " + path.string());

    DataSet data;
    data.positions = readVec3(group.id, "Coordinates");
    data.densities = readFloat(group.id, "Densities");
    data.internalEnergy = readFloat(group.id, "InternalEnergies");
    data.masses = readFloat(group.id, "Masses");
    data.materialIds = readInt32(group.id, "MaterialIDs");
    data.pressures = readFloat(group.id, "Pressures");
    data.smoothingLengths = readFloat(group.id, datasetExists(group.id, "SmoothingLengths") ? "SmoothingLengths" : "SmoothingLength");
    data.velocities = readVec3(group.id, "Velocities");
    data.particleIds = readInt32(group.id, "ParticleIDs");
    if (datasetExists(group.id, "Temperatures")) {
        data.temperatures = readFloat(group.id, "Temperatures");
    } else {
        data.temperatures.assign(data.positions.size(), 0.0f);
    }
    return data;
}

void writeHdf5(const std::filesystem::path& path, const DataSet& data) {
    H5Handle file(H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    if (file.id < 0) throw std::runtime_error("Could not create HDF5 file: " + path.string());
    H5Handle group(H5Gcreate(file.id, "PartType0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    if (group.id < 0) throw std::runtime_error("Could not create PartType0 in " + path.string());

    writeVec3(group.id, "Coordinates", data.positions);
    write1d(group.id, "Densities", H5T_NATIVE_FLOAT, data.densities);
    write1d(group.id, "InternalEnergies", H5T_NATIVE_FLOAT, data.internalEnergy);
    write1d(group.id, "Masses", H5T_NATIVE_FLOAT, data.masses);
    write1d(group.id, "MaterialIDs", H5T_NATIVE_INT32, data.materialIds);
    write1d(group.id, "Pressures", H5T_NATIVE_FLOAT, data.pressures);
    write1d(group.id, "SmoothingLengths", H5T_NATIVE_FLOAT, data.smoothingLengths);
    writeVec3(group.id, "Velocities", data.velocities);
    write1d(group.id, "ParticleIDs", H5T_NATIVE_INT32, data.particleIds);
    write1d(group.id, "Temperatures", H5T_NATIVE_FLOAT, data.temperatures);
}

} // namespace sph
