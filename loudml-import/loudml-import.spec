Name: %{name}
Version: %{version}
Release:	1%{?dist}
Summary:	Data import module for LoudML

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

%install
make -C loudml-import install DESTDIR=%{buildroot}

%files
%defattr(-,root,root,-)
%{_bindir}/*
%{python3_sitelib}/loudml/*
%{python3_sitelib}/loudml_import*.pth
%{python3_sitelib}/loudml_import*.egg-info/*

%doc



%changelog

