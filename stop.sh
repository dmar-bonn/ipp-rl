xhost -local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_mapping_ipp`
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_rotors_simulation`
docker-compose down
