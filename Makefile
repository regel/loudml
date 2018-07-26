BUILD_DIR = $(CURDIR)/build
RPMREPO_DIR := $(BUILD_DIR)/rpmrepo
DEBREPO_DIR := $(BUILD_DIR)/debrepo/stretch

clean:
	$(MAKE) -C public/api clean
	$(MAKE) -C loudml clean
	$(MAKE) -C loudml-elastic clean
	$(MAKE) -C loudml-elastic-aws clean
	$(MAKE) -C loudml-influx clean
	$(MAKE) -C loudml-import clean
	$(MAKE) -C loudml-warp10 clean
	rm -rf build

install:
	$(MAKE) -C public/api install
	$(MAKE) -C loudml install
	$(MAKE) -C loudml-elastic install
	$(MAKE) -C loudml-elastic-aws install
	$(MAKE) -C loudml-influx install
	$(MAKE) -C loudml-import install
	$(MAKE) -C loudml-warp10 install

uninstall:
	$(MAKE) -C loudml-warp10 uninstall
	$(MAKE) -C loudml-influx uninstall
	$(MAKE) -C loudml-import uninstall
	$(MAKE) -C loudml-elastic-aws uninstall
	$(MAKE) -C loudml-elastic uninstall
	$(MAKE) -C loudml uninstall
	$(MAKE) -C public/api uninstall

dev:
	$(MAKE) -C loudml-influx dev
	$(MAKE) -C loudml-import dev
	$(MAKE) -C loudml-elastic dev
	$(MAKE) -C loudml-elastic-aws dev
	$(MAKE) -C loudml-warp10 dev
	$(MAKE) -C loudml dev
	$(MAKE) -C public/api dev

test:
	$(MAKE) -C loudml test
	$(MAKE) -C loudml-elastic test
	$(MAKE) -C loudml-elastic-aws test
	$(MAKE) -C loudml-influx test
	$(MAKE) -C loudml-import test
	$(MAKE) -C loudml-warp10 test

rpm:
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C public/api rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C base rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-elastic rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-elastic-aws rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-influx rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-import rpm
	@echo -e "\nRPM packages:"
	@find $(BUILD_DIR) -name '*.rpm'

deb:
	$(MAKE) -C public/api deb
	mv public/*.deb .
	$(MAKE) -C loudml deb
	$(MAKE) -C base deb
	$(MAKE) -C loudml-elastic deb
	$(MAKE) -C loudml-elastic-aws deb
	$(MAKE) -C loudml-influx deb
	$(MAKE) -C loudml-import deb
	@echo -e "\nDEB packages:"
	@find -name '*.deb'

.PHONY: check_deb
check_deb:
	scripts/check_deb

$(RPMREPO_DIR)/repodata/repomd.xml: rpm
	createrepo $(RPMREPO_DIR)

.PHONY: rpmrepo
rpmrepo: $(RPMREPO_DIR)/repodata/repomd.xml

.PHONY: rpmrepo-archive
rpmrepo-archive: $(BUILD_DIR)/rpmrepo-$(VERSION).tar

$(BUILD_DIR)/rpmrepo-$(VERSION).tar: rpmrepo
	tar -C $(BUILD_DIR) -cvf "$@" rpmrepo

$(DEBREPO_DIR)/Packages.gz:
	mkdir -p $(dir $@)
	cp *.deb $(dir $@)
	cd $(dir $@)/.. \
	&& dpkg-scanpackages `basename $(dir $@)` | gzip > $@

.PHONY: debrepo
debrepo: $(DEBREPO_DIR)/Packages.gz

repo: rpmrepo

.PHONY: docker
docker:
	$(MAKE) -C docker
