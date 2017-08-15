#!/bin/bash -e
# Elasticsearch post-installation script

# System parameters
cat >> /etc/sysctl.conf <<'EOF'
vm.panic_on_oom=1
vm.swappiness=0
kernel.panic=10
EOF
sysctl -p
swapoff -a

# Data partition configuration
cat >> /etc/fstab <<EOF
/dev/vdb    /srv/data/    ext4     defaults     0 0
EOF
mkdir /srv/data

sed -i 's|.*path.data:.*|path.data: /srv/data/elasticsearch|' \
    /etc/elasticsearch/elasticsearch.yml

# JVM settings
cat > /usr/share/elasticsearch/bin/elasticsearch-jvm-pre-exec <<'EOF'
#!/bin/bash
# Official recommendation is to set ES_HEAP_SIZE to 50% of available RAM
# but we want to keep memory for other apps.
es_heap_size_mb=$(free -m|awk '(NR==2){print int($2*3/4)}')

TMP="$(mktemp)"

sed "
s/^-Xms.*/-Xms${es_heap_size_mb}m/
s/^-Xmx.*/-Xmx${es_heap_size_mb}m/
" /etc/elasticsearch/jvm.options > "$TMP"
sync
cat "$TMP" > /etc/elasticsearch/jvm.options
rm "$TMP"
EOF

chmod +x /usr/share/elasticsearch/bin/elasticsearch-jvm-pre-exec

# Service tuning:
# - auto-restart
# - enable memory-locking
# - explicitely set number of open files (actually this is the default)

sed -i 's/.*bootstrap.memory_lock:.*/bootstrap.memory_lock: true/' \
    /etc/elasticsearch/elasticsearch.yml
sed -i 's/.*network.host:.*/network.host: 0.0.0.0/' \
    /etc/elasticsearch/elasticsearch.yml

mkdir -p /etc/systemd/system/elasticsearch.service.d/
cat > /etc/systemd/system/elasticsearch.service.d/elasticsearch.conf <<EOF
[Service]
Restart=always
RestartSec=5
LimitNOFILE=65536
LimitMEMLOCK=infinity
ExecStartPre=/usr/share/elasticsearch/bin/elasticsearch-jvm-pre-exec
EOF

# Disable elasticsearch
# Must be enabled once data partition is available
systemctl disable elasticsearch.service

# Enable blackbox_exporter for elasticsearch monitoring
systemctl enable blackbox_exporter

# Download X-Pack
# Installation instructions:
# https://www.elastic.co/guide/en/x-pack/current/installing-xpack.html#xpack-package-installation
mkdir -p /var/lib/elasticsearch
wget -O /var/lib/elasticsearch/x-pack-5.4.3.zip \
     https://artifacts.elastic.co/downloads/packs/x-pack/x-pack-5.4.3.zip

# Install TensorFlow (CPU)
# Installation instructions: https://www.tensorflow.org/install/install_linux
pkg="tensorflow-1.2.1-cp34-cp34m-linux_x86_64.whl"
url="https://storage.googleapis.com/tensorflow/linux/cpu/$pkg"
yum install -y epel-release
yum install -y python34-pip
pip3 install -U pip
pip3 install --upgrade "$url"
