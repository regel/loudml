Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	LoudML core package

Group: Applications/System
License: Proprietary
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
BuildRequires: systemd
BuildRequires: systemd-units
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd
Requires: python34
Requires: python34-setuptools
Requires: curl
Requires: loudml-api >= 1.4.0
Requires: loudml-base == %{version}
%{?systemd_requires}

# Disable debug package
%define debug_package %{nil}

%package enterprise
Group:    Applications/System
Summary:  LoudML Enterprise exclusive features
Requires: loudml == %{version}

%package internal
Group:    Applications/System
Summary:  LoudML tools for internal usage
Requires: loudml == %{version}

%description


%description enterprise

Additional model: fingerprints.


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
  useradd --comment "LoudML" --gid loudml --no-create-home --system --shell /sbin/nologin loudml
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

# Enable instance checking
sed -i 's/# *DISABLED check_instance()/check_instance()/' loudml/server.py

[ -n $(grep -E "^    check_instance()$" loudml/server.py) ] && \
    echo -e "error: instance checking no present"

make install DESTDIR=%{buildroot}
install -m 0755 -d %{buildroot}/%{_datarootdir}/loudml
install -m 0644 LICENSE %{buildroot}/%{_datarootdir}/loudml/LICENSE

# PYC binary distribution, mv files to pre-PEP-3147 location to be able to
# load modules
for filename in $(find %{buildroot}/%{python3_sitelib}/loudml/__pycache__/ -name "*.cpython-34.pyc") ;
do
	basename=$(basename $filename) ;
	basename="${basename%.cpython-34.pyc}" ;
	mv $filename %{buildroot}/%{python3_sitelib}/loudml/${basename}.pyc ;
done
for filename in $(find %{buildroot}/%{python3_sitelib}/rmn_common/__pycache__/ -name "*.cpython-34.pyc") ;
do
	basename=$(basename $filename) ;
	basename="${basename%.cpython-34.pyc}" ;
	mv $filename %{buildroot}/%{python3_sitelib}/rmn_common/${basename}.pyc ;
done

# LoudML daemon configuration
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml/plugins.d
install -m 0644 examples/config.yml %{buildroot}/%{_sysconfdir}/loudml/config.yml
%{__install} -m 0644 -D systemd/loudmld.service %{buildroot}/%{_unitdir}/loudmld.service
install -m 0775 -d %{buildroot}/%{_sharedstatedir}/loudml

%files
%defattr(-,root,root,-)
%exclude %{python3_sitelib}/loudml/fingerprints.*
# Exclude source .py files, and PEP3147 __pycache__
%exclude %{python3_sitelib}/loudml/*.py
%exclude %{python3_sitelib}/loudml/__pycache__
%exclude %{python3_sitelib}/rmn_common/*.py
%exclude %{python3_sitelib}/rmn_common/__pycache__
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

# LoudML daemon configuration
%attr(2777,root,loudml) %dir %{_sysconfdir}/loudml
%attr(2777,root,loudml) %dir %{_sysconfdir}/loudml/plugins.d
%config(noreplace) %{_sysconfdir}/loudml/config.yml
%{_unitdir}/loudmld.service
%attr(2775,loudml,loudml) %{_sharedstatedir}/loudml


%files enterprise
%{python3_sitelib}/loudml/fingerprints.pyc


%files internal
%{_bindir}/loudml-lic


%doc



%changelog

