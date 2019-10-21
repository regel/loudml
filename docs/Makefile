all:
	docker run --user $$(id -u):$$(id -g) \
		-v $(PWD):/opt sourcey/spectacle spectacle -t /opt/public /opt/swagger.yml

.PHONY: all
