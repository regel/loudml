# note: macros %%python3_pkgversion, %%python3_other_pkgversion and %%with_python3_other are defined 
# in the minimal buildroot; %%python3_pkgversion is also available in Fedora, so it's possible to have a common
# specfile for EPEL and Fedora

%global srcname bonsai
%global sum A ML wizard REST API

Name:           %{srcname}
Version:        %{version}
Release:        1%{?dist}
Summary:        %{sum}
License:        Proprietary
URL:            http://www.redmintnetwork.com
Source0:        %{srcname}-%{version}.tar.gz

BuildArch:      noarch
#BuildRequires:  python2-devel
BuildRequires:  python%{python3_pkgversion}-devel

%description
An python module to control, train, and run inference
with ML models.

#%package -n python2-%{srcname}
#Summary:        %{sum}
#%{?python_provide:%python_provide python2-%{srcname}}

#%description -n python2-%{srcname}


%package -n python%{python3_pkgversion}-%{srcname}
Summary:        %{sum}
%{?python_provide:%python_provide python%{python3_pkgversion}-%{srcname}}

%description -n python%{python3_pkgversion}-%{srcname}
python%{python3_pkgversion} build of %{srcname}.

%prep
%autosetup

%build
#%py2_build
%py3_build

%install
# Must do the python3_other install first, then python3 and then python2.
# The scripts in /usr/bin are overwritten with every setup.py install.
%py3_install
# PYC binary distribution, mv files to pre-PEP-3147 location to be able to load modules
for filename in $(find %{buildroot}/%{python3_sitelib}/%{srcname}/__pycache__/ -name "*.cpython-34.pyc") ;
do
	basename=$(basename $filename) ;
	basename="${basename%.cpython-34.pyc}" ;
	mv $filename %{buildroot}/%{python3_sitelib}/%{srcname}/${basename}.pyc ;
done

#%py2_install

%check
#%{__python2} setup.py test
#%{__python3} setup.py test

# Note that there is no %%files section for the unversioned python module if we are building for several python runtimes
#%files -n python2-%{srcname}
#%license COPYING
#%doc README.rst
#%{python2_sitelib}/*
#%{_bindir}/sample-exec-%{python2_version}

%files -n python%{python3_pkgversion}-%{srcname}
# Exclude source .py files, and PEP3147 __pycache__
%exclude %{python3_sitelib}/%{srcname}/*.py
%exclude %{python3_sitelib}/%{srcname}/__pycache__
%{python3_sitelib}/*
%{_bindir}/bonsaid
%{_bindir}/bonsai_series
%{_bindir}/bonsai_segmap


%changelog


