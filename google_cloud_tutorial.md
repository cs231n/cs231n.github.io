---
layout: page
title: Google Cloud Tutorial
permalink: /gce-tutorial/
---
# Google Cloud Tutorial #

For the class project and assignments, we offer an option to use Google Compute Engine for developing and testing your 
implementations. This tutorial lists the necessary steps of working on the assignments using Google Cloud. 
For each assignment, we will provide you with an image containing the starter code and all dependencies that you need to 
complete the assignment. This tutorial goes through how to set up your own Google Compute Engine (GCE) instance with the 
provided images for the assignments. Each student will have $50 in credit throughout the quarter. When you sign up for the first time, you also receive $300 credits from Google by default. Please try to use the resources judiciously. But if $100 ends up not being enough, we will try to adjust this number as the quarter goes on.

First, if you don't have a Google Cloud account already, create one by going to the [Google Cloud homepage](https://cloud.google.com/?utm_source=google&utm_medium=cpc&utm_campaign=2015-q2-cloud-na-gcp-skws-freetrial-en&gclid=CP2e4PPpiNMCFU9bfgodGHsA1A "Title") and clicking on "Compute." When you get to the next page, click on the blue “TRY IT FREE” button. If you are not logged into gmail, you will see a page that looks like the one below. Sign into your gmail account or create a new one if you do not already have an account. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-launching-screen.png'>
</div>

If you already have a gmail account, it will direct you to a signup page which looks like the following.
<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-for-free.png'>
</div>

Click the appropriate “yes” or “no” button for the first option, and check “yes” for the latter two options after you have read the required agreements. Press the blue “Agree and continue” button to continue to the next page to enter the requested information (your name, billing address and credit card information). Once you have entered the required information, press the blue “Start my free trial” button. You will be greeted by a page like this: 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-dashboard-screen.png'>
</div>

To change the name of your project, click on “Manage project settings” on the “Project info” button and save your changes. 
You will launch virtual instances for assignments using our disk image containing each assignment’s starter code and pre-configured with the environments necessary for each assignment. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-dashboard-screen.png'>
</div>

To launch a virtual instance, go to the “Compute Engine” menu on the left column of your dashboard and click on VM instances.  Then click on the blue "CREATE" button on the next page. This will take you to a page that looks like the screenshot below.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-create-instance-screen.png'>
</div>

Make sure that the Zone is set to be us-west1-b (for assignments where you need to use GPU instances). Under “Machine type” pick the “8 vCPUs” option. Click on the “customize” button under “Machine type” and make sure that the number of cores is set to 8 and the number of GPUs is set to 0 (we will not be using GPUs in assignment 1). Click on the “Change” button under “Boot disk”, choose "OS images", check "Ubuntu 16.04 LTS" and click on the blue "select" button. Check "Allow HTTP traffic" and "Allow HTTPS traffic". Click on "disk" and then "Disks" and uncheck "Delete boot disk when instance is deleted". Click on the blue “Create” button at the bottom of the page. You should have now successfully created a Google Compute Instance, it might take a few minutes to start running. Your screen should look something like the one below. When you want to stop running the instance, click on the blue stop button above. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-started.png'>
</div>

Take note of your \<YOUR-INSTANCE-NAME\>, in this case, my instance name is instance-2. 
Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from [here](https://cloud.google.com/sdk/docs/ "Title") and follow their installation instructions. The easiest way to connect is using the gcloud compute command below. The tool takes care of authentication for you. On OS X, run:

\<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED\>/bin/gcloud compute ssh --zone=us-west 1-b \<YOUR-INSTANCE-NAME\>

See [this page](https://cloud.google.com/compute/docs/instances/connecting-to-instance "Title") for more detailed instructions. You are now ready to work on the assignments on Google Cloud. 

Run the following command to download the current assignment onto your GCE:

wget http://cs231n.stanford.edu/assignments/2016/winter1516_assignmentX.zip where **X** is the assignment number (1, 2 or 3).

run sudo apt-get install unzip

and 

unzip winter1516_assignment**X**.zip to get the contents. You should now see a folder titled assignment**X**. 

# Using Jupyter Notebook with Google Compute Engine # 
Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and use it on your local browser. 

## Getting a Static IP Address ##
Change the Extenal IP address of your GCE instance to be static (see screenshot below). 
<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-external-ip.png'>
</div>

To Do this, click on the 3 line icon next to the "Google Cloud Platform" button on the top left corner of your screen, go to "Networking" and "External IP addresses" (see screenshot below).

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip.png'>
</div>

To have a static IP address, change **Type** from **Ephemeral** to **Static**. Enter your preffered name for your static IP, mine is assignment-1 (see screenshot below). And click on Reserve. Remember to release the static IP address when you are done because according to [this page](https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/ "Title") Google charges a small fee for unused static IPs. **Type** should now be set to **Static**. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip-naming.png'>
</div>

Take note of your Static IP address (circled on the screenshot below). I used 104.196.224.11 for this tutorial.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip-address.png'>
</div>

## Configuring Jupyter Notebook ##
The following instructions are excerpts from [this page](https://haroldsoh.com/2016/04/28/set-up-anaconda-ipython-tensorflow-julia-on-a-google-compute-engine-vm/ "Title") that has more detailed instructions.

On your GCE instance check if you have a Jupyter configuration file:

ls ~/.jupyter/jupyter_notebook_config.py

If it doesn't exist, create one with: 

jupyter notebook --generate-config

You should see an output like:

Writing default config to:\<PATH_TO_CONFIG_FILE\>

where \<PATH_TO_CONFIG_FILE\> is the path to the configuration file. Mine was written to /home/timnitgebru/.jupyter/jupyter_notebook_config.py

Using your favorite editor (vim, emacs etc...) add the following lines to \<PATH_TO_CONFIG_FILE\>:

c = get_config()

c.NotebookApp.ip = '*'

c.NotebookApp.open_browser = False

c.NotebookApp.port = \<PORT-NUMBER\>

I usually use 7000 or 8000 for \<PORT-NUMBER\>

Save and close the file. 

## Launching and connecting to Jupyter Notebook ##
The instructions below assume that you have SSH'd into your GCE instance using the instructions, have already downloaded and unzipped the current assignment folder into assignment**X** (where X is the assignment number), and have successfully configured Jupyter Notebook.

cd into the assignment directory by running the following command:

cd assignment**X** (where X is the assignment number).

Launch Jupyter notebook using:

jupyter-notebook --no-browser --port=\<PORT-NUMBER\> 

Where \<PORT-NUMBER\> is what you wrote in the prior section.

On your local browser, if you go to http://\<YOUR-EXTERNAL-IP-ADDRESS>:\<PORT-NUMBER\>, you should see something like the screen below. My value for \<YOUR-EXTERNAL-IP-ADDRESS\> was 104.196.224.11 as mentioned above. You should now be able to start working on your assignments. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/jupyter-screen.png'>
</div>

To use the same setup again and avoid having to re-download this software, save your disk once you stop running your instance. Next time you start a new instance or res-start this instance, you can then boot from this disk.

Don't forget to stop your instance when you are completely done (by clicking on the stop button at the top of the page showing your instances). 




