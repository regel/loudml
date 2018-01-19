clean:
	$(MAKE) -C loudml clean
	$(MAKE) -C loudml-elastic clean
	$(MAKE) -C loudml-influx clean

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
