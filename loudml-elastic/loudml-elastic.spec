Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	Elasticsearch module for LoudML

Group: Applications/System
License: Proprietary
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
Requires: python34
Requires: loudml == %{version}

# Disable debug package
%define debug_package %{nil}

%description


%prep
%setup -q


%build
make clean

%pre

# Remove old trailing files. Required for users who have installed <=1.3.2
find %{python3_sitelib} \
  -name '*loudml_elastic-1.2.*.egg-info' -o \
  -name '*loudml_elastic-1.3.0.*.egg-info' -o \
  -name '*loudml_elastic-1.3.1.*.egg-info' -o \
  -name '*loudml_elastic-1.3.2.*.egg-info' \
  | xargs rm -rf

%install
make -C loudml-elastic install DESTDIR=%{buildroot}

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
%exclude %{python3_sitelib}/loudml_elastic*.egg-info/requires.txt
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_elastic*.pth
%dir %{python3_sitelib}/loudml_elastic*.egg-info
%{python3_sitelib}/loudml_elastic*.egg-info/*

%doc



%changelog

