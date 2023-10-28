#!/bin/bash

echo "Activating python enviroment"
source .msenv/bin/activate
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Not able to start the  python environment"
fi
