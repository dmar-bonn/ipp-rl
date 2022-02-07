docker-compose down
docker-compose up -d
docker exec -d -it mapping_ipp_framework_mapping_ipp bash -c "python3 main.py"
