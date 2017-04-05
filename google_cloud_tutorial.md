---
layout: page
title: Google Cloud Tutorial
permalink: /gce-tutorial/
---

For the class project and assignments, we offer an option to use Google Compute Engine for developing and testing your 
implementations. This tutorial lists the necessary steps of working on the assignments using Google Cloud. 
For each assignment, we will provide you with an image containing the starter code and all dependencies that you need to 
complete the assignment. This tutorial goes through how to set up your own Google Compute Engine (GCE) instance with the 
provided images for the assignments. **We do not currently distribute Google Cloud credits to CS231N students but you 
are welcome to use this snapshot on your own budget.**

First, if you don't have a Google Cloud account already, create one by going to the [Google Cloud homepage](https://cloud.google.com/?utm_source=google&utm_medium=cpc&utm_campaign=2015-q2-cloud-na-gcp-skws-freetrial-en&gclid=CP2e4PPpiNMCFU9bfgodGHsA1A "Title"), and clicking on the blue “TRY IT FREE” button. If you are not logged into gmail, you will see a page that looks like the one below. Sign into your gmail account or create a new one if you do not already have an account. 

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
You will launch virtual instances for assignments using our disk image containing each assignment’s starter code and pre-configured with the environments necessary for each assignment. To launch a virtual instance, go to the “Compute Engine” menu in the left column of your dashboard and click on VM instances (see screenshot below).

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-dashboard-screen.png'>
</div>

Click on the “CREATE INSTANCE” blue button at the top. You will see a page that looks like the one below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-create-instance-screen.png'>
</div>

Make sure that the Zone is set to be us-west1-b (for assignments where you need to use GPU instances). Under “Machine type” pick the “8 vCPUs” option. Click on the “customize” under “Machine type” and make sure that the number of cores is set to 8 and the number of GPUs is set to however many you need (we will not be using GPUs in assignment 1). Click on the “Change” button under “Boot disk”, choose “Custom images” and check “cs231n-caffe-torch-keras-lasagne” to use our custom image as your boot disk. Click on the blue “Create” button at the bottom of the page. You should have now successfully started a Google Compute Instance, it might take a few minutes to start. Your screen should look something like the one below:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-started.png'>
</div>

Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from [here](https://cloud.google.com/sdk/docs/ "Title") and follow their installation instructions.
To connect to your GCE instance enter the following command: 

\<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED\>/bin/gcloud compute ssh --zone=us-west 1-b \<YOUR-INSTANCE-NAME\>

The assignments will be under a folder XXX. 

# Using Jupyter Notebook with Google Compute Engine # 
Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and use it on your local browser. First ssh into your GCE instance using the instructions above. cd into the assignment directory by running the following command:

cd Assignment-X (where X is the assignment number)

Open a terminal in your GCE instance and launch Jupyter notebook using:

jupyter-notebook --no-browser --port=\<PORT-NUMBER\> 

I usually use 7000 or 8000 for \<PORT-NUMBER\>

Then open another console on your local machine and run the following command:

\<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED\>/bin/gcloud compute ssh --zone=us-west 1-b --ssh-flag=”-D” --ssh-flag=”1080” --ssh-flag=”-N” --ssh-flag=”-n” \<YOUR-INSTANCE-NAME\>

On your local browser, if you go to http://localhost:\<PORT-NUMBER\>, you should see something like the screen below. You should now be able to start working on your assignments.

<div class='fig figcenter fighighlight'>
  <img src='/assets/jupyter-screen.png'>
</div>


