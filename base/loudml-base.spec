Name:          %{name}
Version:       %{version}
Release:       1%{?dist}
Summary:       Loud ML dependencies
Group:         Applications/System
License:       Open Source
URL:           http://www.loudml.com
Requires:      python34
BuildRequires: python34
AutoReqProv:   no

# Disable debug package
%define debug_package %{nil}

%description

Dependencies for Loud ML.


%install
/opt/redmint/bin/pip install -r %{srcdir}/vendor/requirements.txt -t %{buildroot}/%{_libdir}/loudml/vendor
find %{buildroot}/%{_libdir}/loudml/vendor -name __pycache__ | xargs rm -rf

# rpmbuild tries to automatically optimize Python packages, but this fails
# because it tries to compile Python 3 code with Python 2 compiler.
# As a workaround we make sure nothing is done automatically after the
# install scriptlet.
exit 0


%files
%defattr(-,root,root,-)
%dir %{_libdir}/loudml/vendor
%{_libdir}/loudml/vendor/*

%doc


%changelog

