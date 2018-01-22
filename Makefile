BUILD_DIR = $(CURDIR)/build
RPMREPO_DIR := $(BUILD_DIR)/rpmrepo

clean:
	$(MAKE) -C loudml clean
	$(MAKE) -C loudml-elastic clean
	$(MAKE) -C loudml-influx clean
	rm -rf build

install:
	$(MAKE) -C loudml install
	$(MAKE) -C loudml-elastic install
	$(MAKE) -C loudml-influx install

uninstall:
	$(MAKE) -C loudml-influx uninstall
	$(MAKE) -C loudml-elastic uninstall
	$(MAKE) -C loudml uninstall

test:
	$(MAKE) -C loudml test
	$(MAKE) -C loudml-elastic test
	$(MAKE) -C loudml-influx test

rpm:
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-elastic rpm
	$(MAKE) RPMREPO_DIR=$(RPMREPO_DIR) BUILD_DIR=$(BUILD_DIR) -C loudml-influx rpm
	@echo -e "\nRPM packages:"
	@find $(BUILD_DIR) -name '*.rpm'

$(RPMREPO_DIR)/repodata/repomd.xml: rpm
	createrepo $(RPMREPO_DIR)

.PHONY: rpmrepo
rpmrepo: $(RPMREPO_DIR)/repodata/repomd.xml

.PHONY: rpmrepo-archive
rpmrepo-archive: $(BUILD_DIR)/rpmrepo-$(VERSION).tar

$(BUILD_DIR)/rpmrepo-$(VERSION).tar: rpmrepo
	tar -C $(BUILD_DIR) -cvf "$@" rpmrepo

repo: rpmrepo
