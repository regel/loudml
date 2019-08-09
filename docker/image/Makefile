.PHONY: image repo Dockerfile

ifeq ($(repo_src),buildbot)

repo_url := "file:///repo"
s3_src := aws-buildbot/loudml/repo/deb/stretch
awsopts := --region "eu-west-1"
tag := loudml:buildbot

else ifeq ($(repo_src),local)

repo_url := "file:///repo"
tag ?= loudml:local

else ifeq ($(repo_src),release)

repo_url := "http://loudml.s3-website-eu-west-1.amazonaws.com/repo/deb/"
tag ?= loudml/loudml

else ifeq ($(repo_src),staging)

repo_url := "http://loudml-staging.s3-website-eu-west-1.amazonaws.com/repo/deb/"
tag ?= loudml:staging

else

$(error Set repo_src to either 'buildbot', 'local', 'staging' or 'release')

endif

repo_def:="deb [trusted=yes] $(repo_url) stretch main"

ifeq ($(repo_src),buildbot)
image: repo
else ifeq ($(repo_src),local)
image: repo
endif
image: Dockerfile
	echo "$(repo_def)" > loudml-1.x.list
	docker build -t $(tag) .

ifeq ($(gpu),)
Dockerfile: Dockerfile.in
else
requirements.txt.in: ../../base/vendor/requirements.txt.in
	grep -v tensorflow $< > $@

Dockerfile: gpu.Dockerfile.in requirements.txt.in
endif
ifeq ($(repo_src),buildbot)
	cp -f "$<" "$@"
else ifeq ($(repo_src),local)
	cp -f "$<" "$@"
else ifeq ($(repo_src),release)
	grep -v "COPY repo" "$<" > "$@"
else ifeq ($(repo_src),staging)
	grep -v "COPY repo" "$<" > "$@"
endif

repo:
	mkdir -p "$@"
	rm -rf "$@"/*
ifeq ($(repo_src),buildbot)
	aws $(awsopts) s3 sync "s3://$(s3_src)" "$@"
else ifeq ($(repo_src),local)
	cp ../../*.deb "$@"
endif
	mkdir -p "$@/conf"
	cp -f distributions "$@/conf/distributions"
	cp -f options "$@/conf/options"
	for deb in $@/*.deb; do \
	    reprepro -b "$@" includedeb stretch "$$deb"; \
	done


