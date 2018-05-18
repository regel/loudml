Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	LoudML API package

Group: Applications/System
License: General Public License v2
URL: www.loudml.com
Source0: %{name}-%{version}.tar.gz

BuildRequires: python34 python34-pip
Requires: python34

# Disable debug package
%define debug_package %{nil}

%description


%prep
%setup -q


%build
cd api
make clean

%pre

%install
cd api
make install DESTDIR=%{buildroot}

%files
%defattr(-,root,root,-)
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_api-*.egg-info/*
%{python3_sitelib}/loudml_api-*.pth

%doc



%changelog

