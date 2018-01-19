Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	InfluxDB module for LoudML

Group: Applications/System
License: Proprietary
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
Requires: python34
Requires: python34-pip
Requires: loudml

%description


%prep
%setup -q


%build
make clean

%pre
pip3 install influxdb>=5.0.0

%install
make -C loudml-influx install DESTDIR=%{buildroot}

# PYC binary distribution, mv files to pre-PEP-3147 location to be able to
# load modules
for filename in $(find %{buildroot}/%{python3_sitelib}/loudml/__pycache__/ -name "*.cpython-34.pyc") ;
do
	basename=$(basename $filename) ;
	basename="${basename%.cpython-34.pyc}" ;
	mv $filename %{buildroot}/%{python3_sitelib}/loudml/${basename}.pyc ;
done

%files
%defattr(-,root,root,-)
# Exclude source .py files, and PEP3147 __pycache__
%exclude %{python3_sitelib}/loudml/*.py
%exclude %{python3_sitelib}/loudml/__pycache__
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_influx*.pth
%{python3_sitelib}/loudml_influx*.egg-info/*

%doc



%changelog

