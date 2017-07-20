packer := /opt/hashicorp/bin/packer

# Temporary build directory used by Packer. Must not exist and is destroyed
# after creating images.
packer_build_dir := build

.PHONY: vm
elastic: $(tsdir)/elasticsearch.ts

all: elastic

elasticsearch.qcow2: packer_target := elasticsearch
elasticsearch.qcow2: packer_args := -var 'output_dir=$(packer_build_dir)'
elasticsearch.qcow2: elasticsearch.json

elasticsearch.qcow2:
	$(packer) validate $(packer_args) $<
	$(packer) build $(packer_args) $<
	mv -f "$(packer_build_dir)/$(packer_target)" "$@"
	rmdir "$(packer_build_dir)"
