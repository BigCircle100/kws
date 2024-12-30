
basic_cmd="python3 server.py --ip=0.0.0.0 "

log_path="/data/log/log_$(date +%Y-%m-%d).log"
if [ -n "$LOG_LEVEL" ]; then
    basic_cmd="${basic_cmd} --loglevel=\"$LOG_LEVEL\" "
fi

docker_envs=($(env | awk -F= '/^DOCKER_SERVICE/  {sub(/^DOCKER_SERVICE_/, ""); print}'))
for var in "${docker_envs[@]}"; do
    basic_cmd="${basic_cmd} --$var "
    echo $var
done

echo "*****************************************************" >> ${log_path}
echo "   Data: $(date +'%Y-%m-%d %H:%M:%S')" >> ${log_path}
if [ -n "$SERVICE_IMAGE" ]; then
    echo "  Image: ${SERVICE_IMAGE}" >> ${log_path}
fi
echo "Command: ${basic_cmd}" >> ${log_path}
echo "*****************************************************" >> ${log_path}

basic_cmd="${basic_cmd} >> ${log_path} 2>&1"

eval ${basic_cmd}