#!/bin/bash

make

java -jar target/classify-mnist.jar $@
