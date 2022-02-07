docker-compose build
docker run -v ~/.ssh:/tmp/.ssh -v $(pwd):/mapping_ipp_framework mapping_ipp_framework_mapping_ipp ./build_instructions.sh
# docker run -v $(pwd):/mapping_ipp_framework mapping_ipp_framework_mapping_ipp bash -c "LD_LIBRARY_PATH=. python3 /mapping_ipp_framework/setup.py build_ext --inplace"
