BUILD_DIR = $(CURDIR)/build

clean:
	$(MAKE) -C loudml clean
	$(MAKE) -C loudml-elastic clean
	$(MAKE) -C loudml-influx clean
	rm -rf build

install:
	$(MAKE) -C loudml install
	$(MAKE) -C loudml-elastic install
	$(MAKE) -C loudml-influx install

uninstall:
	$(MAKE) -C loudml-influx uninstall
	$(MAKE) -C loudml-elastic uninstall
	$(MAKE) -C loudml uninstall

test:
	$(MAKE) -C loudml test
	$(MAKE) -C loudml-elastic test
	$(MAKE) -C loudml-influx test

rpm:
	$(MAKE) BUILD_DIR=$(BUILD_DIR) -C loudml rpm
	$(MAKE) BUILD_DIR=$(BUILD_DIR) -C loudml-elastic rpm
	$(MAKE) BUILD_DIR=$(BUILD_DIR) -C loudml-influx rpm
