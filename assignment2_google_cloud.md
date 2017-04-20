---
layout: page
title: Google Cloud Tutorial
permalink: /gce-tutorial/
---
# Google Cloud Tutorial (Part 2 With GPUs) #

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

Fill out the required portions of the form and put a value of 1 or 2 in the **Total Number of GPU dies** section. Once you have your quota increase you can just use GPUs (without requesting a quota increase). After you submit the form, you should receive an email approving your quota increase shortly after (I received my email within a minute). If you don't receive your approval within 2 business days, please inform the course staff. Once you have received your quota increase, you can start an instance with a GPU. To do this, while launching a virtual instance as described in the **Launch a Virtual Instance** section [here](http://cs231n.github.io/gce-tutorial/ "title"), select the number of GPUs to be 1 (or 2 if you requested for a quota increase of 2 and you really really need to use 2). As a reminder, you can only use up to the number of GPUs allowed by your quota. Use your GPUs sparingly because they are expensive. See the pricing [here](https://cloud.google.com/compute/pricing#gpus "title").

## Starting Your Instance With Our Provided Disk ##
For the remaining assignments and the project, we provide you with disks containing the necessary software for the assignments and commonly used frameworks for the project. To use our disk, you first need to create your own custom image using our file, and use this custom image as the boot disk for your new VM instance. 

### Creating a Custom Image Using Our Disk ###
To create your custom image using our provided disk, go to **Compute Engine**, then **Images** and click on the blue **Create Image** button at the top of the page. See the screenshot below.
<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-create-image-screenshot.png'>
</div>

Enter your preferred name in the **Name** field. Mine is called **image-2**. Select cloud storage file for **Source**, enter **cs231n-bucket/myimage.tar.gz** as the **Cloud Storage file** and click on the blue **Create** button. See the screenshot below. It will take a few minutes for your image to be created. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-cloud-storage.png'>
</div>

### Starting Your Instance with Your Custom Image ###
To start your instance using our provided disk, go to **VM Instances** and click on **Create Instance** like you have done before. Follow the same procedure that you have used to create an instance as detailed [here](http://cs231n.github.io/gce-tutorial/ "title") but instead of selecting an entry in **OS images** for **Boot disk**, select **Custom images** and the custom image that you created above. Mine is **image-2**. See the screenshot below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-custom-image.png'>
</div>

You should now be able to launch your instance with our custom image. The custom disk is 60GB and uses Ubuntu 16.04 LTS. The default python version in the system is Python 2.7.2 and there is a virtual environment called **myVE35** with version 3.5.2. To use this virtual environment called run 

```
source myVE35/bin/activate
```

The disk should also have Jupyter **X**, CUDA 8.0, CUDNN 5.1, Pytorch 0.1.11_5 and Tensorflow 1.0.1.


To work on assignment 2, **MORE INSTRUCTIONS ON THE FOLDER STRUCTURE ETC...***


