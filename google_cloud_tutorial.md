---
layout: page
title: Google Cloud Tutorial
permalink: /gce-tutorial/
---
# Google Cloud Tutorial #

## BEFORE WE BEGIN ##
### BIG REMINDER: Make sure you stop your instances! ###
(We know you won't read until the very bottom once your assignment is running, so we are printing this at the top too since it is ***super important***)

Don't forget to ***stop your instance*** when you are done (by clicking on the stop button at the top of the page showing your instances), otherwise you will ***run out of credits*** and that will be very sad. :( 

If you follow our instructions below correctly, you should be able to restart your instance and the downloaded software will still be available.

<div class='fig figcenter fighighlight'>
  <img src='/assets/sadpuppy_nocredits.png'>
</div>


## Create and Configure Your Account ##

For the class project and assignments, we offer an option to use Google Compute Engine for developing and testing your 
implementations. This tutorial lists the necessary steps of working on the assignments using Google Cloud. **We expect this tutorial to take about an hour. Don't get intimidated by the steps, we tried to make the tutorial detailed so that you are less likely to get stuck on a particular step. Please tag all questions related to Google Cloud with google_cloud on Piazza.**

This tutorial goes through how to set up your own Google Compute Engine (GCE) instance to work on the assignments. Each student will have $100 in credit throughout the quarter. When you sign up for the first time, you also receive $300 credits from Google by default. Please try to use the resources judiciously. But if $100 ends up not being enough, we will try to adjust this number as the quarter goes on.

First, if you don't have a Google Cloud account already, create one by going to the [Google Cloud homepage](https://cloud.google.com/?utm_source=google&utm_medium=cpc&utm_campaign=2015-q2-cloud-na-gcp-skws-freetrial-en&gclid=CP2e4PPpiNMCFU9bfgodGHsA1A "Title") and clicking on **Compute**. When you get to the next page, click on the blue **TRY IT FREE** button. If you are not logged into gmail, you will see a page that looks like the one below. Sign into your gmail account or create a new one if you do not already have an account. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-launching-screen.png'>
</div>

Click the appropriate **yes** or **no** button for the first option, and check **yes** for the second option after you have read the required agreements. Press the blue **Agree and continue** button to continue to the next page to enter the requested information (your name, billing address and credit card information). Remember to select "**Individual**" as "Account Type":

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-register-info.png'>
</div>

Once you have entered the required information, press the blue **Start my free trial** button. You will be greeted by a page like this: 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-welcome-screen.png'>
</div>

Press the "Google Cloud Platform" (in red circle), and it will take you to the main dashboard:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-dashboard-screen.png'>
</div>

To change the name of your project, click on [**Go to project settings**](console.cloud.google.com/iam-admin/settings/project) under the **Project info** section.

## Create an image from our provided disk ##

For all assignments and the final project, we provide you with a pre-configured disk that contains the necessary environment and deep learning frameworks. To use our disk, you first need to create your own custom image using our file, and use this custom image as the boot disk for your new VM instance.  

Go to **Compute Engine**, then **Images** and click on the blue **Create Image** button at the top of the page. See the screenshot below.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-create-image.png'>
</div>

Enter your preferred name in the **Name** field. Mine is called **cs231n-image**. Select **Cloud Storage file** for **Source**, enter **cs231n-repo/deep-ubuntu.tar.gz** and click on the blue **Create** button. See the screenshot below. It will take a few minutes for your image to be created (about 10-15 in our experience, though your mileage may vary). 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-config-image.png'>
</div>

## Launch a Virtual Instance ##

To launch a virtual instance, go to the **Compute Engine** menu on the left column of your dashboard and click on **VM instances**. 

Then click on the blue **Create** button on the next page. This will take you to a page that looks like the screenshot below. **(NOTE: Please carefully read the instructions in addition to looking at the screenshots. The instructions tell you exactly what values to fill in).**

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-create-instance.png'>
</div>

Make sure that the Zone is set to be **us-west1-b** (especially for assignments where you need to use GPU instances). Under **Machine type** pick the **8 vCPUs** option. Click on the **customize** button under **Machine type** and make sure that the number of cores is set to 8 and the number of GPUs is set to **None** (we will not be using GPUs in assignment 1. GPU will be covered later in this tutorial). 

Click on the **Change** button under **Boot disk**, choose **Custom images**, you will see this screen:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-select-image.png'>
</div>

Select the image you created in the previous step, here it's **cs231n-image**. Also increase the boot disk size as you see fit. Click **Select** and you will get back to the "create instance" screen.

Check **Allow HTTP traffic** and **Allow HTTPS traffic**. Expand the **Management, disks, networking, SSH keys** menu if it isn't visible, select **Disks** tab, and uncheck **Delete boot disk when instance is deleted**.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-check-https.png'>
</div>

Click on the blue **Create** button at the bottom of the page. You should have now successfully created a Google Compute Instance, it might take a few minutes to start running. When the instance is ready, your screen should look something like the one below. When you want to stop running the instance, click on the blue stop button above. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-started.png'>
</div>

Take note of your instance name, you will need it to ssh from your laptop. 

## Connect to Your Virtual Instance ##

Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from [here](https://cloud.google.com/sdk/docs/ "Title") and follow their installation instructions. **NOTE: this tutorial assumes that you have performed step #4 on the website which they list as optional**. When prompted, make sure you select `us-west1-b` as the time zone. 

The easiest way to connect is using the gcloud compute command below. The tool takes care of authentication for you. On your laptop (OS X for example), run:

```
gcloud compute ssh --zone=us-west1-b <YOUR-INSTANCE-NAME>
```

If `gcloud` command is not in system path, you can also reference it by its full path `/<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED>/bin/gcloud`. See [this page](https://cloud.google.com/compute/docs/instances/connecting-to-instance "Title") for more detailed instructions. 

## First time setup ##

Upon your first ssh, you need to run a one-time setup script and reload the `.bashrc` to activate the libraries. The exact command is

```
/home/shared/setup.sh && source ~/.bashrc
```

The command will download a git repo, patch your `.bashrc` and copy a jupyter notebook config file to your home directory. If you ever switch account/username, you will have to re-run the setup command. If you see any permission error, simply prepend `sudo` to the command. 

When the command finishes without error, run `which python` on the command line and it should report `/home/shared/anaconda3/bin/python`. See screenshot:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-setup-script.png'>
</div>

(don't worry about the Tensorflow warning message)

Our provided image supports the following frameworks:

- [Anaconda3](https://www.anaconda.com/what-is-anaconda/), a python package manager. You can think of it as a better alternative to `pip`. 
- Numpy, matplotlib, and tons of other common scientific computing packages.
- [Tensorflow 1.7](https://www.tensorflow.org/), both CPU and GPU. 
- [PyTorch 0.3](https://www.pytorch.org/), both CPU and GPU. 
- [Keras](https://keras.io/) that works with Tensorflow 1.7
- [Caffe2](https://caffe2.ai/), CPU only. Note that it is very different from the original Caffe. 
- Nvidia runtime: CUDA 9.0 and cuDNN 7.0. They only work when you create a Cloud GPU instance, which we will cover later.  

The `python` on our image is `3.6.4`, and has all the above libraries installed. It should work out of the box for all assignments unless noted otherwise. You don't need `virtualenv`, but if you insist, Anaconda has [its own way](https://conda.io/docs/user-guide/tasks/manage-environments.html). If you need libraries not mentioned above, you can always run `conda install <mylib>` yourself. 

You are now ready to work on the assignments on Google Cloud!


## Using Jupyter Notebook with Google Compute Engine ##
Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and connect to it with your local browser. 

### Getting a Static IP Address ###
Change the Extenal IP address of your GCE instance to be static (see screenshot below). 
<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-external-ip.png'>
</div>

To Do this, click on the 3 line icon next to the **Google Cloud Platform** button on the top left corner of your screen, go to **VPC network** and **External IP addresses** (see screenshot below).

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip.png'>
</div>

To have a static IP address, change **Type** from **Ephemeral** to **Static**. Enter your prefered name for your static IP, ours is `cs231n-ip` (see screenshot below). And click on Reserve. Remember to release the static IP address when you are done because according to [this page](https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/ "Title") Google charges a small fee for unused static IPs. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip-naming.png'>
</div>

Take note of your Static IP address (circled on the screenshot below). We use 35.185.240.182 for this tutorial.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-external-ip-address.png'>
</div>

### Adding a Firewall rule ###

One last thing you have to do is adding a new firewall rule allowing TCP acess to a particular port number. The default port we use for Jupyter is **7000**. You can find this default value in the config file generated at setup time (`~/.jupyter/jupyter_notebook_config.py`). Feel free to change it.

Click on the 3-line icon at the top of the page next to **Google Cloud Platform**. On the menu that pops up on the left column, go to **VPC network** and **Firewall rules** (see the screenshot below). 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-firewall-rule.png'>
</div>

Click on the blue **CREATE FIREWALL RULE** button. Enter whatever name you want: we use cs231n-rule. Select "All instances in the network" for **Targets** (if the menu item exists). Enter `0.0.0.0/0` for **Source IP ranges** and `tcp:<port-number>` for **Specified protocols and ports** where `<port-number>` is the number you used above. Click on the blue **Create** button. See the screenshot below.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-firewall-rule-create.png'>
</div>
 

### Launching and connecting to Jupyter Notebook ###

After you ssh into your GCE instance using the prior instructions, run Jupyter notebook from the folder with your assignment files. As a quick example, let's launch it from `/home/shared` folder.

```
cd /home/shared
jupyter-notebook --no-browser --port=7000
```

If you simply run `jupyter-notebook` without any command line arguments, it will pick up the default config values in `~/.jupyter/jupyter_notebook_config.py`. In our disk image, it is `no-browser` and port 7000 by default.

The command should block your stdin and display something like:

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-jupyter-console.png'>
</div>

The important line (underscored in red) has the token for you to login from laptop. Replace the "localhost" part with your external IP address created in prior steps. In our example, the URL should be 

```
http://35.185.240.182:7000/?token=aad408a5bcc56f8a7d79db4e144507537e4cf927bd1ab6bc
```

If there is no token, simply go to `http://35.185.240.182:7000`. 

If you visit the above URL on your local browser, you should see something like the screen below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-jupyter-screen.png'>
</div>

## Submission: Transferring Files From Your Instance To Your Computer ##

When you are done with your assignments, run the submission script in your assignment folder to make a zip file. Please refer to specific instructions for each assignment.  

Once you create the zip file, e.g. `assignment1.zip`, you will transfer the file from GCE instance to your local laptop. There is an [easy command](https://cloud.google.com/sdk/gcloud/reference/compute/scp) for this purpose:

```
gcloud compute scp <user>@<instance-name>:/path/to/assignment1.zip /local/path
```

For example, to download files from our instance to the current folder:

```
gcloud compute scp tonystark@cs231:/home/shared/assignment1.zip .
```

The transfer works in both directions. To upload a file to GCE:

```
gcloud compute scp /my/local/file tonystark@cs231:/home/shared/
```

Another (perhaps easier) option proposed by a student is to directly download the zip file from Jupyter. After running the submission script and creating assignment1.zip, you can download that file directly from Jupyter. To do this, go to Jupyter Notebook and click on the zip file, which will be downloaded to your local computer. 

## BIG REMINDER: Make sure you stop your instances! ##

Don't forget to stop your instance when you are done (by clicking on the stop button at the top of the page showing your instances). You can restart your instance and the downloaded software will still be available. 

We have seen students who left their instances running for many days and ran out of credits. You will be charged per hour when your instance is running. This includes code development time. We encourage you to read up on Google Cloud, regularly keep track of your credits and not solely rely on our tutorials.
