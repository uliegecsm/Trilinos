set -ex

mkdir -p build
cd build

cp ../cmakePresetTrilinos.json ../CMakePresets.json
cd ..

cmake -S . --preset=gcc-OpenMP-arm64 -DCMAKE_INSTALL_PREFIX=/workspaces/trilinos-install

# cmake \
#   -DCMAKE_INSTALL_PREFIX=/Users/florentdistree/trilinos-install \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_OSX_ARCHITECTURES="arm64" \
#   -DTPL_ENABLE_MPI=OFF \
#   -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
#   -DTrilinos_ENABLE_Tpetra=ON \
#   -DTrilinos_ENABLE_Teuchos=ON \
#   -DTrilinos_ENABLE_Belos=ON \
#   -DTrilinos_ENABLE_Ifpack2=ON \
#   -DTrilinos_ENABLE_MueLu=ON \
#   -DTrilinos_ENABLE_Amesos2=ON \
#   -DTrilinos_ENABLE_Zoltan2=ON \
#   -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
#   -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
#   -DTpetra_INST_INT_INT=ON \
#   -DMueLu_ENABLE_Experimental=ON \
#   -DXpetra_ENABLE_Experimental=ON \
#   -DTrilinos_SET_GROUP_AND_PERMISSIONS_ON_INSTALL_BASE_DIR=/Users/florentdistree \
#   -DAmesos2_ENABLE_KLU2=ON \
#   -DTrilinos_ENABLE_Galeri=ON \
#   -DTPL_ENABLE_SuperLU=OFF ..

cmake --build --preset=gcc-OpenMP-arm64 --target=install -j5