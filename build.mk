BUILD_DIR ?= $(CURDIR)/build
RPMBUILD_DIR ?= $(BUILD_DIR)/rpmbuild
rpmsrc_dir := $(RPMBUILD_DIR)/SOURCES

# TODO: try to find the right path
# The line below works because the file is only included from subdirectories.
TOP_DIR ?= .

VERSION ?= $(shell $(TOP_DIR)/scripts/version)

export LOUDML_VERSION=$(VERSION)

FULLNAME := $(NAME)-$(VERSION)

# setup.py install options

INSTALL_OPTS ?=

ifneq ($(DESTDIR),)
	INSTALL_OPTS += "--root=$(DESTDIR)"
endif

# Build RPM source archive
define rpmsrc
@echo -e "  RPMSRC\t$(rpmsrc_dir)/$(1).tar.gz"
@[ -n "${VIRTUAL_ENV}" ] || exit 0 && echo -e "error: do not build RPM from a virtualenv!" && exit 1
@mkdir -p "$(rpmsrc_dir)"
@set -x && cd $(TOP_DIR) && git ls-files | xargs tar -czf /tmp/$(1).tar.gz \
    --transform "s|^|$(1)/|"
@mv -f /tmp/$(1).tar.gz "$(rpmsrc_dir)/"
endef

# Build RPM package
define rpmbuild
@echo -e "  RPMBUILD\t$(1)"
rpmbuild --define "name $(2)" \
         --define "version $(VERSION)" \
         --define "_topdir $(RPMBUILD_DIR)" \
         --define "_rpmdir $(RPMREPO_DIR)" \
   	     $(3) \
         -bb "$<"
endef

debian/changelog: debian/changelog.in
	sed 's/@VERSION@/$(VERSION)/' $< > $@

# Build DEB package
define debbuild
@echo -e "  DEBBUILD\t$(1)"
debuild --preserve-envvar=LOUDML_VERSION -us -uc -b
endef

