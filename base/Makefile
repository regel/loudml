NAME := loudml-base

PIP_COMPILE ?= pip-compile

vendor/requirements.txt: vendor/requirements.txt.in
	$(PIP_COMPILE) -U -o $@ $<
	echo "https://github.com/regel/python-warp10client.git" >> $@

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec vendor/requirements.txt
	$(call rpmbuild,$(FULLNAME),$(NAME),--define "srcdir $(CURDIR)")

deb: $(NAME).deb

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

clean:
	rm -rf build

.PHONY: rpm deb debian/changelog debian/control

include ../build.mk
