Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	Data import module for Loud ML

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
  -name '*loudml_import-1.2.*.egg-info' -o \
  -name '*loudml_import-1.3.0.*.egg-info' -o \
  -name '*loudml_import-1.3.1.*.egg-info' -o \
  -name '*loudml_import-1.3.2.*.egg-info' \
  | xargs rm -rf

%install
make -C loudml-import install DESTDIR=%{buildroot}

%files
%defattr(-,root,root,-)
%{_bindir}/*
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_import*.pth
%dir %{python3_sitelib}/loudml_import*.egg-info
%{python3_sitelib}/loudml_import*.egg-info/*

%doc



%changelog

