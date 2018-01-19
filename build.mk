DESTDIR ?= /
BUILD_DIR ?= $(CURDIR)/build
rpmbuild_dir := $(BUILD_DIR)/rpmbuild
rpmsrc_dir := $(rpmbuild_dir)/SOURCES
repo_dir := $(BUILD_DIR)/repo

VERSION ?= $(shell git describe --tags --match 'v*.*.*' | \
                       sed -e 's/^v//' -e 's/-/./g')

FULLNAME := $(NAME)-$(VERSION)

# Build RPM source archive
define rpmsrc
@echo -e "  RPMSRC\t$(rpmsrc_dir)/$(1).tar.gz"
@mkdir -p "$(rpmsrc_dir)"
@set -x && cd $(CURDIR)/.. && git ls-files | xargs tar -czf /tmp/$(1).tar.gz \
    --transform "s|^|$(1)/|"
@mv -f /tmp/$(1).tar.gz "$(rpmsrc_dir)/"
endef

# Build RPM package
define rpmbuild
@echo -e "  RPMBUILD\t$(1)"
rpmbuild --define "name $(2)" \
         --define "version $(VERSION)" \
         --define "_topdir $(rpmbuild_dir)" \
         --define "_rpmdir $(repo_dir)" \
         -bb "$<"
endef

