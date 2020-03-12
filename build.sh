#!/bin/sh

set -x

#protoc protos/*.proto --python_out=. || exit -1
protoc_3.3/bin/protoc protos/*.proto --python_out=. || exit -1

if [ ! -d "tensorflow_models" ]; then
  git clone "https://github.com/tensorflow/models.git" "tensorflow_models"
  ln -s "tensorflow_models/research/object_detection" .
fi

exit 0
