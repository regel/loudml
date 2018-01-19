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
Requires: python34-pip
Requires: python34-yaml
Requires: curl

%description


%prep
%setup -q


%build
make clean

%pre
pip3 install dateutils>=0.6.6
pip3 install Flask>=0.12.2
pip3 install Flask-restful>=0.3.6
pip3 install tensorflow==1.3.0
pip3 install h5py==2.7.1
pip3 install hyperopt==0.1
pip3 install requests>=2.17.0


%install
cd loudml
make install DESTDIR=%{buildroot}

# PYC binary distribution, mv files to pre-PEP-3147 location to be able to
# load modules
for filename in $(find %{buildroot}/%{python3_sitelib}/loudml/__pycache__/ -name "*.cpython-34.pyc") ;
do
	basename=$(basename $filename) ;
	basename="${basename%.cpython-34.pyc}" ;
	mv $filename %{buildroot}/%{python3_sitelib}/loudml/${basename}.pyc ;
done

# LoudML daemon configuration
install -m 0755 -d %{buildroot}/%{_sysconfdir}/loudml
install -m 0644 examples/config.yml %{buildroot}/%{_sysconfdir}/loudml/config.yml
%{__install} -m 0644 -D systemd/loudmld.service %{buildroot}/%{_unitdir}/loudmld.service
install -m 0755 -d %{buildroot}/%{_sharedstatedir}/loudml

%files
%defattr(-,root,root,-)
# Exclude source .py files, and PEP3147 __pycache__
%exclude %{python3_sitelib}/loudml/*.py
%exclude %{python3_sitelib}/loudml/__pycache__
%{_bindir}/*
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml-*.egg-info/*

# LoudML daemon configuration
%{_sysconfdir}/loudml/config.yml
%{_unitdir}/loudmld.service
%{_sharedstatedir}/loudml

%doc



%changelog

