---
layout: page
title: Google Cloud Tutorial Part 2 (with GPUs)
permalink: /gce-tutorial-gpus/
---
# Google Cloud Tutorial (Part 2 With GPUs) #
This tutorial assumes that you have already gone through the first Google Cloud tutorial for assignment 1 [here](http://cs231n.github.io/gce-tutorial/ "title"). The first tutorial takes you through the process of setting up a Google Cloud account, launching a VM instance, accessing Jupyter Notebook from your local computer, working on assignment 1 on your VM instance, and transferring files to your local computer. While you created a VM instance without a GPU in the first tutorial, this one walks you through the necessary steps to create an instance with a GPU, and use our provided disk images to work on assignment 2. If you haven't already done so, we advise you to go through the first tutorial to be comfortable with the process of creating an instance with the right configurations and accessing Jupyter Notebook from your local computer.

## Changing your Billing Account ##
Everyone enrolled in the class should have received $100 Google Cloud credits by now. Some of you might want to use these instead of your free trial credits. To do this, follow the instructions on [this website](https://support.google.com/cloud/answer/6293499?hl=en "Title") to change the billing address associated with your project to **CS 231n- Convolutional Neural Netwks for Visual Recog-Set 1**.
	 
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

Once you have your quota increase you can just use GPUs (without requesting a quota increase). After you submit the form, you should receive an email approving your quota increase shortly after (I received my email within a minute). If you don't receive your approval within 2 business days, please inform the course staff. Once you have received your quota increase, you can start an instance with a GPU. To do this, while launching a virtual instance as described in the **Launch a Virtual Instance** section [here](http://cs231n.github.io/gce-tutorial/ "title"), select the number of GPUs to be 1 (or 2 if you requested for a quota increase of 2 and you really really need to use 2). As a reminder, you can only use up to the number of GPUs allowed by your quota. Use your GPUs sparingly because they are expensive. See the pricing [here](https://cloud.google.com/compute/pricing#gpus "title").

## Starting Your Instance With Our Provided Disk ##
For the remaining assignments and the project, we provide you with disks containing the necessary software for the assignments and commonly used frameworks for the project. To use our disk, you first need to create your own custom image using our file, and use this custom image as the boot disk for your new VM instance. 

### Creating a Custom Image Using Our Disk ###
To create your custom image using our provided disk, go to **Compute Engine**, then **Images** and click on the blue **Create Image** button at the top of the page. See the screenshot below.
<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-create-image-screenshot.png'>
</div>

Enter your preferred name in the **Name** field. Mine is called **final-cs231n**. Select cloud storage file for **Source**, enter **cs231n-files/cs231n_image.tar.gz** as the **Cloud Storage file** and click on the blue **Create** button. See the screenshot below. It will take a few minutes for your image to be created (about 10-15 in our experience, though your mileage may vary). 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-cloud-storage.png'>
</div>

### Starting Your Instance with Your Custom Image ###
To start your instance using our provided disk, go to **VM Instances** and click on **Create Instance** like you have done before. Make sure you start the instance in **us-west1-b**. Follow the same procedure that you have used to create an instance as detailed [here](http://cs231n.github.io/gce-tutorial/ "title") but with the following differences:

Make sure to provision 1 GPU to your instance by clicking "Customize" in the "Machine Type" box, as in the screenshot below:

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-instance-gpus.png'>
</div>

Instead of selecting an entry in **OS images** for **Boot disk**, select **Custom images** and the custom image that you created above. Mine is **final-cs231n**. See the screenshot below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-custom-image.png'>
</div>

It should take about 5 minutes for the instance to get created. You should now be able to launch your instance with our custom image. The custom disk is 40GB and uses Ubuntu 16.04 LTS. 
The default python version in the system is Python 2.7.2 and there is a virtual environment in **/home/cs231n/myVE35** with version 3.5.2. 

You don't need to create a new virtual environment for this assignment -- we are providing you with one, with all the Python packages you need for the assignment already installed. So, unlike the previous assignment, you do not need to run any commands to create the virtual environment (and install packages, etc), just one command to activate it. To use this virtual environment, run the following command (you may have to do this every time you start your instance up, if you don't see your bash prompt prefaced by "(myVE35)"): 

```
source /home/cs231n/myVE35/bin/activate
```

Here's what you should see after running that to confirm you're in the virtual environment:

```
username@instance-2:~$ source /home/cs231n/myVE35/bin/activate
(myVE35) username@instance-2:~$ which python
/home/cs231n/myVE35/bin/python
(myVE35) username@instance-2:~$ 
```

The disk should also have Jupyter 1.0.0, CUDA 8.0, CUDNN 5.1, Pytorch 0.1.11_5 and TensorFlow 1.0.1. GPU support should be automatically enabled for PyTorch and TensorFlow. 

### Getting started on Assignment 2 ###

Get back to the Assignment 2 instructions [here](http://cs231n.github.io/assignments2017/assignment2/).

## Transferring Files to Your Local Computer ##
After following assignment 2 instructions to run the submission script and create assignment2.zip, you can download that file directly from Jupyter. To do this, go to Jupyter Notebook and click on the zip file (in this case assignment2.zip). The file will be downloaded to your local computer. You can also use the **gcloud compute copy-files** command to transfer files as discussed in the **Submission: Transferring Files From Your Instance To Your Computer** section in [the first GCE tutorial](http://cs231n.github.io/gce-tutorial/ "title").

# BIG REMINDER: Make sure you stop your instances! #

Don't forget to stop your instance when you are done (by clicking on the stop button at the top of the page showing your instances). You can restart your instance and the downloaded software will still be available.
