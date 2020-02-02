# Cmd run via COPR service:
# make -f .copr/Makefile srpm outdir="<outdir>" spec"loudml.spec"

BUILD_DIR = $(CURDIR)/build
TOP_DIR = $(CURDIR)/.
NAME := loudml
RPMBUILD_DIR ?= $(BUILD_DIR)/rpmbuild
rpmsrc_dir := $(RPMBUILD_DIR)/SOURCES

all: srpm

prepare:
	dnf -y install dnf-plugins-core git tito
	dnf builddep "$(spec)"

srpm: prepare
	git config user.email "copr@loudml.io"
	git config user.name "COPR Agent"
	tito tag --accept-auto-changelog \
		--use-release=`git describe --tags --match 'v*.*.*' | sed -e 's/.*-\([0-9]*\)-.*/\1/g'`
	tito build --offline --tgz -o "$(rpmsrc_dir)"
	rpmbuild \
		--define "_topdir $(RPMBUILD_DIR)" \
		--define "_srcrpmdir $(outdir)" \
		-bs "$(spec)"

.PHONY: prepare srpm
