NAME := loudml-elastic

install:
	./setup.py install $(INSTALL_OPTS)

uninstall:
	pip3 uninstall -y loudml-elastic

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

.PHONY: rpm

include ../build.mk
