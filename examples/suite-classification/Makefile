SOURCES := $(shell find . -name "*.clj*")


all: uberjar

uberjar: target/classify-mnist.jar

target/classify-mnist.jar: $(SOURCES)
	lein clean && lein uberjar
