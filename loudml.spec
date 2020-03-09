%global srcname loudml
# Turn off Python bytecode compilation to reduce package size
# See #manual-bytecompilation on docs.fedoraproject.org
%undefine __brp_python_bytecompile
%global __python %{__python3}
%global __os_install_post %(echo '%{__os_install_post}' | sed -e 's!/usr/lib[^[:space:]]*/brp-python-bytecompile[[:space:]].*$!!g')

Name: loudml
Version: 1.5.0
Release:	1%{?dist}
Summary:	Loud ML core package

Group: Applications/System
License: MIT
URL: www.loudml.io

# Source is created by:
# git clone %%url
# tito build --tgz --tag %%name-%%version-%%release
Source0:    %name-%version.tar.gz

BuildRequires: git
BuildRequires: gcc
BuildRequires: python3-devel
BuildRequires: python3-pip
BuildRequires: python3-rpm-macros
BuildRequires: systemd
BuildRequires: systemd-units
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd
Requires: python3-setuptools
%{?systemd_requires}
AutoReqProv:   no

# Disable debug package
%define debug_package %{nil}

%description


%prep
%setup -q


%build
make clean
%py3_build

%pre
if ! getent group loudml; then
  groupadd --system loudml
fi
if ! getent passwd loudml; then
  useradd --comment "Loud ML" --gid loudml --no-create-home --system --shell /sbin/nologin loudml
fi

%post
%systemd_post loudmld.service

%preun
%systemd_preun loudmld.service

%postun
%systemd_postun_with_restart loudmld.service

%install

PYTHONUSERBASE=%{buildroot}/opt/venvs/loudml/ \
	pip3 install --user -r requirements.txt .[cpu]
PYTHONUSERBASE=%{buildroot}/opt/venvs/loudml/ \
	%py3_install
find

install -m 0755 -d %{buildroot}/%{_datarootdir}/loudml
install -m 0644 LICENSE %{buildroot}/%{_datarootdir}/loudml/LICENSE

# Loud ML daemon configuration
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml/plugins.d
install -m 0644 examples/config.yml %{buildroot}/%{_sysconfdir}/loudml/config.yml
%{__install} -m 0644 -D systemd/loudmld.service %{buildroot}/%{_unitdir}/loudmld.service
install -m 0775 -d %{buildroot}/%{_sharedstatedir}/loudml
cp -r templates %{buildroot}/%{_sharedstatedir}/loudml

exit 0  # Prevent .so file strip causing libhdf5-5773eb11.so.103.0.0: ELF load command address/offset not properly aligned
# Reference: https://www.redhat.com/archives/rpm-list/2005-March/msg00086.html

%files
%defattr(-,root,root,-)
%{_bindir}/*
%license %{_datarootdir}/loudml/LICENSE
%attr(2775,loudml,loudml) /opt/venvs/loudml/
%{python3_sitelib}/%{srcname}-*.egg-info/
%{python3_sitelib}/%{srcname}/

# Loud ML daemon configuration
%attr(2777,root,loudml) %{_sysconfdir}/loudml/
%config(noreplace) %{_sysconfdir}/loudml/config.yml
%{_unitdir}/loudmld.service
%attr(2775,loudml,loudml) %{_sharedstatedir}/loudml/


%doc



%changelog
* Sun Feb 02 2020 Sebastien Leger <sebastien.regel@gmail.com>
- new package built with tito


