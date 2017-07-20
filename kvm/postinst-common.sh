#!/bin/bash -e
# Common post-installation script

# Setup basic network configuration
cat > /etc/sysconfig/network-scripts/ifcfg-eth0 <<EOF
NAME=eth0
DEVICE=eth0
ONBOOT=yes
NETBOOT=yes
IPV6INIT=yes
BOOTPROTO=dhcp
TYPE=Ethernet
EOF

# Console on serial port (for "virsh console")
systemctl enable getty@ttyS0

# Enable node_exporter for system monitoring
systemctl enable node_exporter

yum install -y yum-plugin-versionlock 
yum versionlock elasticsearch
yum update -y
yum clean all
# Disable all updates once QCOW2 image is generated
echo "exclude=*" >> /etc/yum.conf

# Cleanup
rm -f /tmp/*.sh
