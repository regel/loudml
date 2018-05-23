.PHONY: image repo

s3_src := aws-buildbot/loudml/repo/deb/stretch
awsopts := --region "eu-west-1"

tag := loudml:buildbot

image:
	docker build -t $(tag) .

repo:
	mkdir -p "$@"
	rm -rf "$@"/*
	aws $(awsopts) s3 sync "s3://$(s3_src)" "$@"
	mkdir -p "$@/conf"
	cp -f distributions "$@/conf/distributions"
	cp -f options "$@/conf/options"
	for deb in $@/*.deb; do \
	    reprepro -b "$@" includedeb stretch "$$deb"; \
	done


