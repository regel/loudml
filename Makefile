BUILD_DIR = $(CURDIR)/build
RPMREPO_DIR := $(BUILD_DIR)/rpmrepo
DEBREPO_DIR := $(BUILD_DIR)/debrepo/stretch

NAME := loudml
unittests ?= $(addprefix tests/, \
	test_config.py test_metrics.py test_misc.py test_model.py test_schemas.py \
	test_base.py test_memdatasource.py test_donut.py)

install:
	python3 setup.py install $(INSTALL_OPTS)

uninstall:
	pip3 uninstall -y loudml

clean:
	python3 setup.py clean
	rm -rf build dist

dev:
	python3 setup.py develop --no-deps

test:
	nosetests -v tests/

coverage:
	nosetests --with-coverage \
            -v $(unittests)

unittest:
	nosetests -v $(unittests)


$(NAME).rpm: $(NAME).spec
	$(call rpmsrc,$(FULLNAME))
	$(call rpmbuild,$(FULLNAME),$(NAME))

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

.PHONY: rpm deb debian/changelog debian/control

include build.mk

rpm: $(NAME).rpm
	@echo -e "\nRPM packages:"
	@find $(BUILD_DIR) -name '*.rpm'

deb: $(NAME).deb
	@echo -e "\nDEB packages:"
	@find $(BUILD_DIR) -name '*.deb'

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

.PHONY: fmt
fmt:
	find loudml -type f -name "*.py" | xargs autopep8 -i
