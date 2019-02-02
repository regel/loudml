BUILD_DIR = $(CURDIR)/build
RPMREPO_DIR := $(BUILD_DIR)/rpmrepo
DEBREPO_DIR := $(BUILD_DIR)/debrepo/stretch

clean:
	$(MAKE) -C loudml clean
	rm -rf build

install:
	$(MAKE) -C loudml install

uninstall:
	$(MAKE) -C loudml uninstall

dev:
	$(MAKE) -C loudml dev

test:
	$(MAKE) -C loudml test

rpm:
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C base rpm
	@echo -e "\nRPM packages:"
	@find $(BUILD_DIR) -name '*.rpm'

deb:
	mv public/*.deb .
	$(MAKE) -C loudml deb
	$(MAKE) -C base deb
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
