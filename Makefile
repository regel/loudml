
JAVA_HOME ?= /usr/lib/jvm/$(rpm -i java-1.8.0-openjdk -q --queryformat=%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH})

VERSION ?= $(shell git describe --tags --match 'v*.*.*' | \
                       sed -e 's/^v//' -e 's/-/./g')

prod:
	export JAVA_HOME=$(JAVA_HOME)
	$(MAKE) -C java/
	exit 0

clean:
	exit 0

install:
	exit 0

uninstall:
	exit 0

BUILD_DIR ?= $(CURDIR)/build
rpmbuild_dir := $(BUILD_DIR)/rpmbuild
rpmsrc_dir := $(rpmbuild_dir)/SOURCES

# Build RPM source archive
define rpmsrc
@echo -e "  RPMSRC\t$(rpmsrc_dir)/$(1).tar.gz"
@mkdir -p "$(rpmsrc_dir)"
@git ls-files | xargs tar -czf /tmp/$(1).tar.gz \
    --transform "s|^|$(1)/|"
@mv -f /tmp/$(1).tar.gz "$(rpmsrc_dir)/"
endef


.PHONY: prod

