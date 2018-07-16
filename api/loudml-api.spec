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

# Remove old trailing files. Required for users who have installed <1.3.3
find %{python3_sitelib} \
  -name '*loudml_api-1.3.0.*.egg-info' \
  | xargs rm -rf

%install
cd api
make install DESTDIR=%{buildroot}

%files
%defattr(-,root,root,-)
%{python3_sitelib}/loudml/*
%dir %{python3_sitelib}/loudml_api-*.egg-info
%{python3_sitelib}/loudml_api-*.egg-info/*
%{python3_sitelib}/loudml_api-*.pth

%doc



%changelog

