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

# To simplify the host deployment, only one SSH host key (ECDSA) will be
# generated. Disable the others.
rm -f /etc/ssh/ssh_host_{dsa,rsa,ed25519}_key{,.pub}
sed -e 's|^\(HostKey /etc/ssh/ssh_host_rsa_key\)|#\1|' \
    -e 's|^\(HostKey /etc/ssh/ssh_host_dsa_key\)|#\1|' \
    -e 's|^\(HostKey /etc/ssh/ssh_host_ed25519_key\)|#\1|' \
    -i /etc/ssh/sshd_config

# Cleanup
rm -f /tmp/*.sh
