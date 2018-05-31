NAME := loudml-base

vendor/requirements.txt: vendor/requirements.txt.in
	pip-compile -U -o $@ $<

rpm: $(NAME).rpm

$(NAME).rpm: $(NAME).spec vendor/requirements.txt
	$(call rpmbuild,$(FULLNAME),$(NAME),--define "srcdir $(CURDIR)")

deb: $(NAME).deb

$(NAME).deb: debian/changelog
	$(call debbuild,$(FULLNAME))

.PHONY: rpm deb debian/changelog debian/control

include ../build.mk
