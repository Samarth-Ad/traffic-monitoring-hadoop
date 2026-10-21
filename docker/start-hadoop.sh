#!/bin/bash
set -e

service ssh start

if [ ! -f /hadoop-data/namenode/current/VERSION ]; then
    echo "Formatting HDFS NameNode..."
    hdfs namenode -format -force
fi

echo "Checking HDFS..."
if ! jps | grep -q NameNode; then
    start-dfs.sh
fi

echo "HDFS status:"
jps

echo "Container is ready."
exec tail -f /dev/null
