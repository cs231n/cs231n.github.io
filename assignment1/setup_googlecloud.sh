#!/usr/bin/env bash

# This is the set-up script for Google Cloud.
sudo apt-get update
sudo apt-get install libncurses5-dev
sudo apt-get install python-dev
sudo apt-get install python-pip
sudo apt-get install libjpeg8-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
pip install pillow
sudo apt-get build-dep python-imaging
sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
sudo pip install virtualenv  
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
deactivate
echo "**************************************************"
echo "*****  End of Google Cloud Set-up Script  ********"
echo "**************************************************"
echo ""
echo "If you had no errors, You can proceed to work with your virtualenv as normal."
echo "(run 'source .env/bin/activate' in your assignment directory to load the venv,"
echo " and run 'deactivate' to exit the venv. See assignment handout for details.)"
