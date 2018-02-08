NAME := loudml-influx

install:
	./setup.py install $(INSTALL_OPTS)

uninstall:
	pip3 uninstall -y loudml-influx

clean:
	./setup.py clean
	rm -rf build

dev:
	./setup.py develop --no-deps

test:
	nosetests-3.4 -v tests/

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec
	$(call rpmsrc,$(FULLNAME))
	$(call rpmbuild,$(FULLNAME),$(NAME))

deb: $(NAME).deb

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

.PHONY: rpm deb

include ../build.mk
