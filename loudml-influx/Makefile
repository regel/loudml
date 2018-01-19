NAME := loudml-influx

install:
	./setup.py install --root=$(DESTDIR)

uninstall:
	pip3 uninstall -y loudml-influx

clean:
	./setup.py clean
	rm -rf build

test:
	nosetests-3.4 -v tests/

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec
	$(call rpmsrc,$(FULLNAME))
	$(call rpmbuild,$(FULLNAME),$(NAME))

.PHONY: rpm

include ../build.mk
