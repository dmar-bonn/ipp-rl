docker-compose down
docker-compose up -d
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_mapping_ipp`
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_rotors_simulation`
docker exec -it mapping_ipp_framework_mapping_ipp bash -c "python3 main.py"
