Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	Elasticsearch AWS module for Loud ML

Group: Applications/System
License: Proprietary
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
Requires: python34
Requires: loudml == %{version}, loudml-elastic == %{version}

# Disable debug package
%define debug_package %{nil}

%description

%prep
%setup -q

%build
make clean

%pre

%install
make -C loudml-elastic-aws install DESTDIR=%{buildroot}

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
# Skip dependencies management by pkg_resources (does not work well with our
# vendor system)
%exclude %{python3_sitelib}/loudml_elastic_aws*.egg-info/requires.txt
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_elastic_aws*.pth
%dir %{python3_sitelib}/loudml_elastic_aws*.egg-info
%{python3_sitelib}/loudml_elastic_aws*.egg-info/*

%doc



%changelog

