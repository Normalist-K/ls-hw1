#!/bin/bash
source ../config.sh

cd $BASE_DIR
mkdir tools/

cd ./tools
# download opensmile
wget https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz

# extract under ./tools
tar -zxvf opensmile-3.0-linux-x64.tar.gz