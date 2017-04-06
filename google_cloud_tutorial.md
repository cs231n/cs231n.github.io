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

Make sure that the Zone is set to be us-west1-b (for assignments where you need to use GPU instances). Under “Machine type” pick the “8 vCPUs” option. Click on the “customize” button under “Machine type” and make sure that the number of cores is set to 8 and the number of GPUs is set to however many you need (we will not be using GPUs in assignment 1). Click on the “Change” button under “Boot disk”, choose “Custom images” and check “cs231n-caffe-torch-keras-lasagne” to use our custom image as your boot disk. Click on the blue “Create” button at the bottom of the page. You should have now successfully started a Google Compute Instance, it might take a few minutes to start. Your screen should look something like the one below:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-started.png'>
</div>

Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from [here](https://cloud.google.com/sdk/docs/ "Title") and follow their installation instructions.

Before you can ssh into your GCE instance using the command line, you first need to generate a new SSH key-pair and apply the public key to your project. Follow instructions under "Generating a new SSH key-pair" on [this page](https://cloud.google.com/compute/docs/instances/connecting-to-instance "Title") to generate an SSH key-pair. The process for Linux and OS X machines is copied and pasted from the website below:

To generate a new SSH key-pair on Linux or OSX workstations:

1. Open a terminal on your workstation and use the ssh-keygen command to generate a new key-pair. Specify the -C flag to add a comment with your Google username. The example creates a private key named my-ssh-key, and a public key file named my-ssh-key.pub.

ssh-keygen -t rsa -f ~/.ssh/my-ssh-key -C [USERNAME]

where [USERNAME] is the user on the instance for whom you will apply the key. If the user does not exist on the instance, Compute Engine automatically creates it using the username that you specify in this command.

2. Restrict access to your my-ssh-key private key so that only you can read it and nobody can write to it.

chmod 400 ~/.ssh/my-ssh-key

3. Go to the metadata page for your project.

You do this by clicking on the "Metadata" tab on the left column

4. Click SSH Keys to show a list of project-wide public SSH keys.

5. Click the Edit button so that you can modify the public SSH keys in your project.

6. Obtain the contents of the ~/.ssh/my-ssh-key.pub public key file with the cat command.

cat ~/.ssh/my-ssh-key.pub

The terminal shows your public key in the following form:

ssh-rsa [KEY_VALUE] [USERNAME]
where:

[KEY_VALUE] is the generated public key value.

[USERNAME] is your username.

7. Copy the output from the cat command and paste it as a new item in the list of SSH keys.
At the bottom of the SSH Keys page, click **Save** to save your new project-wide SSH key.

The public key is now set to work across all of the instances in your project. 

Now, to connect to your GCE instance through SSH enter the following command: 

\<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED\>/bin/gcloud compute ssh --zone=us-west 1-b \<YOUR-INSTANCE-NAME\>

The assignments will be under a folder \<TODO\>/assignment**X**. where **X** is the assignment number (1, 2 or 3).

# Using Jupyter Notebook with Google Compute Engine # 
Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and use it on your local browser. First ssh into your GCE instance using the instructions above. cd into the assignment directory by running the following command:

cd assignment**X** (where X is the assignment number)

Open a terminal in your GCE instance and launch Jupyter notebook using:

jupyter-notebook --no-browser --port=\<PORT-NUMBER\> 

I usually use 7000 or 8000 for \<PORT-NUMBER\>

Then open another console on your local machine and run the following command:

\<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED\>/bin/gcloud compute ssh --zone=us-west 1-b --ssh-flag=”-D” --ssh-flag=”1080” --ssh-flag=”-N” --ssh-flag=”-n” \<YOUR-INSTANCE-NAME\>

On your local browser, if you go to http://localhost:\<PORT-NUMBER\>, you should see something like the screen below. You should now be able to start working on your assignments.

<div class='fig figcenter fighighlight'>
  <img src='/assets/jupyter-screen.png'>
</div>


