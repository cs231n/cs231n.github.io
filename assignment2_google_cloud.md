---
layout: page
title: Google Cloud Tutorial
permalink: /gce-tutorial/
---
# Google Cloud Tutorial (Part 2 With GPUs) #

## Changing your Billing Account ##
Everyone enrolled in the class should have recieved $100 Google Cloud credits by now. Some of you might want to use these instead of your free trial credits. To do this, follow the instructions on [this website](https://support.google.com/cloud/answer/6293499?hl=en "Title") to change the billing address associated with your project to **CS 231n- Convolutional Neural Netwks for Visual Recog-Set 1**.
	 
## Requesting GPU Quota Increase ##
To start an instance with a GPU (for the first time) you first need to request an increase in the number of GPUs you can use. To do this, go to your console, click on the **Computer Engine** button and select the **Quotas** menu. You will see a page that looks like the one below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-quotas-screen.png'>
</div>

Click on the blue **Request Increase** button. This opens a new tab with a long form titled **Google Compute Engine Quota Change Request Form** (see screenshot below). 
<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-quotas-form.png'>
</div>

Fill out the required portions of the form and put a value of 1 or 2 in the **Total Number of GPU dies** section. Once you have your quota increase you can just use GPUs (without requesting a quota increase). After you submit the form, you should recieve an email approving your quota increase shortly after (I recieved my email within a minute). If you don't receive your approval within 2 business days, please inform the course staff. 


## Starting Your Instance With Our Provided Disk ##
For the remaining assignments and the project, we provide you a disk which contains the necessary software for the assignments. To use our disk, you first need to create your own custom disk using our file and use this custom image as the boot disk for your new VM instance. 

### Creating a Custom Image Using Our Disk ###



To start your instance using our provided disk, go to **VM Instances** and click on **Create Instance** like you have done below. However, instead of selecting 


### Starting Your Instance with Our Custom Image ###
