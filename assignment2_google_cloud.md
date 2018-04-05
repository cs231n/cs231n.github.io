---
layout: page
title: Google Cloud Tutorial Part 2 (with GPUs)
permalink: /gce-tutorial-gpus/
---
# Google Cloud Tutorial (Part 2 With GPUs) #
This tutorial assumes that you have already gone through the first Google Cloud tutorial for assignment 1 [here](http://cs231n.github.io/gce-tutorial/ "title"). The first tutorial takes you through the process of setting up a Google Cloud account, launching a VM instance, accessing Jupyter Notebook from your local computer, working on assignment 1 on your VM instance, and transferring files to your local computer. While you created a VM instance without a GPU in the first tutorial, this one walks you through the necessary steps to create an instance with a GPU, and use our provided disk images to work on assignment 2. If you haven't already done so, we advise you to go through the first tutorial to be comfortable with the process of creating an instance with the right configurations and accessing Jupyter Notebook from your local computer.

## Changing your Billing Account ##
Everyone enrolled in the class should have received $100 Google Cloud credits by now. In order to use GPUs, you have to use these coupons instead of your free trial credits. To do this, follow the instructions on [this website](https://support.google.com/cloud/answer/6293499?hl=en "Title") to change the billing address associated with your project to **CS 231n- Convolutional Neural Netwks for Visual Recog-Set 1**.
	 
## Requesting GPU Quota Increase ##
To start an instance with a GPU (for the first time) you first need to request an increase in the number of GPUs you can use. To do this, go to your console, click on the **Computer Engine** button and select the **Quotas** menu. You will see a page that looks like the one below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-quotas-screen.png'>
</div>

Click on the blue **Request Increase** button. This opens a new tab with a long form titled **Google Compute Engine Quota Change Request Form** (see screenshot below). 
<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-quotas-form.png'>
</div>

Fill out the required portions of the form and put a value of **1** in the **Total Number of GPU dies** section. You only need to do this for the **us-west1** region... don't request GPUs anywhere else, since all your instances should also live in us-west1. 

Once you have your quota increase you can just use GPUs (without requesting a quota increase). After you submit the form, you should receive an email approving your quota increase shortly after (I received my email within a minute). If you don't receive your approval within 2 business days, please inform the course staff. Once you have received your quota increase, you can start an instance with a GPU. To do this, while launching a virtual instance as described in the **Launch a Virtual Instance** section [here](http://cs231n.github.io/gce-tutorial/ "title"), select the number of GPUs to be 1 (or 2 if you requested for a quota increase of 2 and you really really need to use 2). As a reminder, you can only use up to the number of GPUs allowed by your quota. 

**NOTE: Use your GPUs sparingly because they are expensive. See the pricing [here](https://cloud.google.com/compute/pricing#gpus "title").** For example, in assignment 2, you only need to have a GPU instance for question 5. So you can work on all other parts of the assignment with a CPU instance that does *not* have any GPUs. Once you are done with the questions that do not require a GPU, you can transfer your assignment zip file to your local computer as discussed in the **Transferring Files to Your Local Computer** section below. When you get to question 5, start a GPU instance and transfer your zip file from your local computer to your instance. Refer to [this page](https://cloud.google.com/compute/docs/instances/transfer-files "title") for details on transferring files to Google Cloud. 

## Starting Your Instance With Our Provided Disk ##
DONE

### Starting Your Instance with Your Custom Image ###
DONE
<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-instance-gpus.png'>
</div>
DONE

**NOTE:** Some students have reported GPU instances whose drivers disappear upon restarts. And there is strong evidence to suggest that the issue happens when Ubuntu auto-installs security updates upon booting up. To disable auto-updating, run

```
sudo apt-get remove unattended-upgrades
```

after logging into your instance for the first time.

### Load the virtual environment ###

DONE 

### Getting started on Assignment 2 ###
As in assignment 1, you can download the assignment zip file by running:

```
wget http://cs231n.stanford.edu/assignments/2017/spring1617_assignment2.zip
```

You can unzip the assignment zip file by running:

```
sudo apt-get install zip #For when you need to zip the file to submit it.
sudo apt-get install unzip
unzip spring1617_assignment2.zip
```

Get back to the Assignment 2 instructions [here](http://cs231n.github.io/assignments2017/assignment2/). Follow the instructions starting from the **Download data** section.

**NOTE: Some students have seen errors saying that an NVIDIA driver is not installed.** If you see this error, follow the instructions [here](https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script "title") to run the script for **Ubuntu 16.04 LTS or 16.10 - CUDA 8 with latest driver** under **Installing GPU drivers**. I.e. copy and paste the script into a file, lets call the file install_cuda.sh. And then run 

```
sudo bash install_cuda.sh 
```

Once you run the script above, run the commands 
```
nvidia-smi
nvcc --version
```

to ensure that the drivers are installed.

