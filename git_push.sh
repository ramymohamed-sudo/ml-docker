#!/bin/bash

hostname=$(hostname)
# echo $hostname
# echo "$PWD"

if [ $hostname == 'k8s-virtual-machine' ]
then
    echo "This is the analytical server";
    git add .;
    git commit -m "$(date)";
    git push origin master;

else
    echo "This is a Raspberry-PI ";
    git pull origin master;
    chmod 755 ./auto-switches/web_flask.py
    # python3 cycle2_0x.py;

fi


# DIR="/home/k8s/minikube/github-scripts/Telit-and-BG96/BG96-final"
# if [ -d "$DIR" ]; then
#   echo "Installing config files in ${DIR}..."
# fi

# try -e, for loop, while loop, search in string, awk and grep