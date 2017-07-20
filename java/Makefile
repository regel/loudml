
### Sanity checks
ifeq ($(JAVA_HOME),)
$(error JAVA_HOME not set - check Java install and export JAVA_HOME)
# JAVA_HOME=/usr/lib/jvm/<your jvm>/
# PATH=$$JAVA_HOME/bin:$$PATH
endif

all: source zip

source:
	python3 gen_device_search.py > \
		native/src/main/java/org/elasticsearch/examples/nativescript/script/DeviceSearchScriptFactory.java

### Output binary. Version must match exactly Elasticsearch version
### native/build/distributions/beat-native-script-5.4.1.0.zip
zip:
	( cd native/ ; ./gradlew assemble )

.PHONY: zip source

