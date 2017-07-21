install
cdrom
lang en_US.UTF-8
keyboard fr
unsupported_hardware
network --bootproto=dhcp --hostname=elasticsearch
rootpw root
firewall --disabled
#selinux --permissive
selinux --disabled
timezone UTC
unsupported_hardware
bootloader --location=mbr
bootloader --append="net.ifnames=0 biosdevname=0 console=tty0 console=ttyS0,9600n8" --timeout=3
text
skipx
zerombr
clearpart --all --initlabel
autopart
auth --enableshadow --passalgo=sha512 --kickstart
firstboot --disabled
eula --agreed
services --enabled=NetworkManager,sshd
reboot

repo --name=base --baseurl=http://mirror.centos.org/centos/$releasever/os/$basearch/
repo --name=updates --baseurl=http://mirror.centos.org/centos/$releasever/updates/$basearch/
repo --name=extras --baseurl=http://mirror.centos.org/centos/$releasever/extras/$basearch/

repo --name=elasticsearch-5.x --baseurl=https://artifacts.elastic.co/packages/5.x/yum

repo --name=prometheus-rpm_release --baseurl=https://packagecloud.io/prometheus-rpm/release/el/7/$basearch

%packages --excludedocs
# common packages
@Core
biosdevname
curl
epel-release
iotop
net-tools
node_exporter
ntp
sudo
tcpdump
tmux
vim-enhanced
wget
yum-utils

# image specific packages
java-1.8.0-openjdk-headless
elasticsearch-5.4.3
blackbox_exporter

# unnecessary firmware
-aic94xx-firmware
-atmel-firmware
-b43-openfwwf
-bfa-firmware
-ipw2100-firmware
-ipw2200-firmware
-ivtv-firmware
-iwl100-firmware
-iwl1000-firmware
-iwl3945-firmware
-iwl4965-firmware
-iwl5000-firmware
-iwl5150-firmware
-iwl6000-firmware
-iwl6000g2a-firmware
-iwl6050-firmware
-libertas-usb8388-firmware
-ql2100-firmware
-ql2200-firmware
-ql23xx-firmware
-ql2400-firmware
-ql2500-firmware
-rt61pci-firmware
-rt73usb-firmware
-xorg-x11-drv-ati-firmware
-zd1211-firmware


# qlogic firmwares
-iwl*-firmware
-aic*-firmware
-ivtv-firmware

%end


%post

%end
