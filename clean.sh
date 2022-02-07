rm -rf src/
docker run -v $(pwd):/mapping_ipp_framework mapping_ipp_framework_mapping_ipp bash -c "catkin clean -y"
