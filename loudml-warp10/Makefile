NAME := loudml-warp10

install:
	./setup.py install $(INSTALL_OPTS)

uninstall:
	pip3 uninstall -y loudml-warp10

clean:
	./setup.py clean
	rm -rf build

dev:
	./setup.py develop --no-deps

test:
	# WARP10_READ_TOKEN and WARP10_WRITE_TOKEN environment variables must be set
	nosetests-3.4 -v tests/

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec
	$(call rpmsrc,$(FULLNAME))
	$(call rpmbuild,$(FULLNAME),$(NAME))

deb: $(NAME).deb

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

.PHONY: rpm deb debian/changelog debian/control

include ../build.mk
