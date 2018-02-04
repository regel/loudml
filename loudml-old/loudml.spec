# note: macros %%python3_pkgversion, %%python3_other_pkgversion and %%with_python3_other are defined 
# in the minimal buildroot; %%python3_pkgversion is also available in Fedora, so it's possible to have a common
# specfile for EPEL and Fedora

%global srcname loudml_old
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
BuildRequires: python34 python34-pip
BuildRequires: systemd
BuildRequires: systemd-units

%description
An python module to control, train, and run inference
with ML models.

#%package -n python2-%{srcname}
#Summary:        %{sum}
#%{?python_provide:%python_provide python2-%{srcname}}

#%description -n python2-%{srcname}


%package -n python%{python3_pkgversion}-%{srcname}
Summary:        %{sum}
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd
Requires: python34
Requires: python34-pip
Requires: python34-yaml
Requires: curl

%{?python_provide:%python_provide python%{python3_pkgversion}-%{srcname}}

%description -n python%{python3_pkgversion}-%{srcname}
python%{python3_pkgversion} build of %{srcname}.

%prep
%autosetup

%build
#%py2_build
%py3_build

%install
%{__install} -m 0755 -d %{buildroot}/%{_sbindir}
%{__install} -m 0644 -D lib/systemd/%{srcname}.service %{buildroot}/%{_unitdir}/%{srcname}.service 
%{__install} -m 0644 -D etc/sysconfig/%{srcname} %{buildroot}/%{_sysconfdir}/sysconfig/%{srcname} 
%{__install} -m 0644 -D etc/%{srcname}.template.json %{buildroot}/%{_sysconfdir}/%{srcname}/%{srcname}.template.json 
%{__install} -m 0644 -D etc/%{srcname}-anomalies.template.json %{buildroot}/%{_sysconfdir}/%{srcname}/%{srcname}-anomalies.template.json 

# Must do the python3_other install first, then python3 and then python2.
# The scripts in /usr/bin are overwritten with every setup.py install.
%py3_install

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

%post
%systemd_post %{srcname}.service
systemctl daemon-reload

%preun
%systemd_preun %{srcname}.service

%postun
%systemd_postun_with_restart %{srcname}.service


%files -n python%{python3_pkgversion}-%{srcname}
%{python3_sitelib}/*
%{_bindir}/loudmld_old
%{_bindir}/loudml_times
%{_bindir}/loudml_ivoip
%{_unitdir}/%{srcname}.service
%config %{_sysconfdir}/sysconfig/*
%config %{_sysconfdir}/%{srcname}


%changelog


