NAME := loudml
unittests ?= $(addprefix tests/, \
	test_config.py test_metrics.py test_misc.py test_model.py test_schemas.py \
	test_server.py)

install:
	./setup.py install $(INSTALL_OPTS)

uninstall:
	pip3 uninstall -y loudml

clean:
	./setup.py clean
	rm -rf build dist

dev:
	./setup.py develop --no-deps

test:
	nosetests-3.4 -v tests/

coverage:
	-nosetests-3.4 --with-coverage \
            -v $(unittests)

unittest:
	nosetests -v $(unittests)

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec
	$(call rpmsrc,$(FULLNAME))
	$(call rpmbuild,$(FULLNAME),$(NAME))

deb: $(NAME).deb

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

.PHONY: rpm deb debian/changelog debian/control

include ../build.mk
