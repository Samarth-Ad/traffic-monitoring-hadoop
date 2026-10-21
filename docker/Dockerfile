FROM eclipse-temurin:11-jdk-jammy

ENV HADOOP_VERSION=3.4.2
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

ENV HDFS_NAMENODE_USER=root
ENV HDFS_DATANODE_USER=root
ENV HDFS_SECONDARYNAMENODE_USER=root

RUN apt-get update && \
    apt-get install -y wget openssh-server openssh-client sudo procps net-tools && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://dlcdn.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xzf hadoop-${HADOOP_VERSION}.tar.gz -C /opt && \
    mv /opt/hadoop-${HADOOP_VERSION} ${HADOOP_HOME} && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

RUN mkdir -p /hadoop-data/namenode \
             /hadoop-data/datanode \
             /hadoop-data/tmp \
             /run/sshd \
             /root/.ssh && \
    chmod 700 /root/.ssh

COPY core-site.xml ${HADOOP_CONF_DIR}/core-site.xml
COPY hdfs-site.xml ${HADOOP_CONF_DIR}/hdfs-site.xml
COPY mapred-site.xml ${HADOOP_CONF_DIR}/mapred-site.xml
COPY yarn-site.xml ${HADOOP_CONF_DIR}/yarn-site.xml

RUN echo "export JAVA_HOME=/opt/java/openjdk" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

RUN ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 && \
    cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    printf "Host localhost\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n" > /root/.ssh/config && \
    chmod 600 /root/.ssh/config

EXPOSE 9870 9864 8088 8042 9000

COPY start-hadoop.sh /start-hadoop.sh
RUN chmod +x /start-hadoop.sh

CMD ["/start-hadoop.sh"]
