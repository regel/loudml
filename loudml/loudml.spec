Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	Loud ML core package

Group: Applications/System
License: Proprietary
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
BuildRequires: python34-devel
BuildRequires: systemd
BuildRequires: systemd-units
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd
Requires: python34
Requires: python34-setuptools
Requires: curl
Requires: loudml-base == %{version}
%{?systemd_requires}

# Disable debug package
%define debug_package %{nil}

%package internal
Group:    Applications/System
Summary:  Loud ML tools for internal usage
Requires: loudml == %{version}

%description


%description internal

Internal tools:
- license management


%prep
%setup -q


%build
make clean

%pre
if ! getent group loudml; then
  groupadd --system loudml
fi
if ! getent passwd loudml; then
  useradd --comment "Loud ML" --gid loudml --no-create-home --system --shell /sbin/nologin loudml
fi

# Remove old trailing files. Required for users who have installed <=1.3.2
find %{python3_sitelib} \
  -name '*loudml-1.2.*.egg-info' -o \
  -name '*loudml-1.3.0.*.egg-info' -o \
  -name '*loudml-1.3.1.*.egg-info' -o \
  -name '*loudml-1.3.2.*.egg-info' \
  | xargs rm -rf

%post
%systemd_post loudmld.service

%preun
%systemd_preun loudmld.service

%postun
%systemd_postun_with_restart loudmld.service

%install
cd loudml

make install DESTDIR=%{buildroot}
install -m 0755 -d %{buildroot}/%{_datarootdir}/loudml
install -m 0644 LICENSE %{buildroot}/%{_datarootdir}/loudml/LICENSE

# Loud ML daemon configuration
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml/plugins.d
install -m 0644 examples/config.yml %{buildroot}/%{_sysconfdir}/loudml/config.yml
%{__install} -m 0644 -D systemd/loudmld.service %{buildroot}/%{_unitdir}/loudmld.service
install -m 0775 -d %{buildroot}/%{_sharedstatedir}/loudml
cp -r templates %{buildroot}/%{_sharedstatedir}/loudml


%files
%defattr(-,root,root,-)
# Skip dependencies management by pkg_resources (does not work well with our
# vendor system)
%exclude %{python3_sitelib}/loudml-*.egg-info/requires.txt
%exclude %{_bindir}/loudml-lic
%{_bindir}/*
%license %{_datarootdir}/loudml/LICENSE
%dir %{python3_sitelib}/rmn_common
%{python3_sitelib}/rmn_common/*
%{python3_sitelib}/loudml/*
%dir %{python3_sitelib}/loudml-*.egg-info
%{python3_sitelib}/loudml-*.egg-info/*

# Loud ML daemon configuration
%attr(2777,root,loudml) %dir %{_sysconfdir}/loudml
%attr(2777,root,loudml) %dir %{_sysconfdir}/loudml/plugins.d
%config(noreplace) %{_sysconfdir}/loudml/config.yml
%{_unitdir}/loudmld.service
%attr(2775,loudml,loudml) %{_sharedstatedir}/loudml
%attr(2775,loudml,loudml) %{_sharedstatedir}/loudml/templates


%files internal
%{_bindir}/loudml-lic


%doc



%changelog

