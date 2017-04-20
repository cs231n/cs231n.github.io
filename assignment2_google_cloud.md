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

Enter your preferred name in the **Name** field. Mine is called **final-cs231n**. Select cloud storage file for **Source**, enter **cs231n-files/cs231n_image.tar.gz** as the **Cloud Storage file** and click on the blue **Create** button. See the screenshot below. It will take a few minutes for your image to be created (about 10-15 in our experience, though your mileage may vary). 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-cloud-storage.png'>
</div>

### Starting Your Instance with Your Custom Image ###
To start your instance using our provided disk, go to **VM Instances** and click on **Create Instance** like you have done before. Follow the same procedure that you have used to create an instance as detailed [here](http://cs231n.github.io/gce-tutorial/ "title") but instead of selecting an entry in **OS images** for **Boot disk**, select **Custom images** and the custom image that you created above. Mine is **final-cs231n**. See the screenshot below. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/google-cloud-select-custom-image.png'>
</div>

You should now be able to launch your instance with our custom image. The custom disk is 40GB and uses Ubuntu 16.04 LTS. The default python version in the system is Python 2.7.2 and there is a virtual environment called **myVE35** with version 3.5.2. The python in this virtual environment already has all the Python packages you'll need for this assignment installed. To use this virtual environment, run the following command (you'll have to do this every time you start your instance up): 

```
source /home/cs231n/myVE35/bin/activate
```

The disk should also have Jupyter **X**, CUDA 8.0, CUDNN 5.1, Pytorch 0.1.11_5 and TensorFlow 1.0.1. GPU support should be automatically enabled for PyTorch and TensorFlow. 

### Getting started on Assignment 2 ###

To work on assignment 2, download the code from the following zip file, and unzip it. 

### Download data:
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `assignment2` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

### Start IPython:
After you have the CIFAR-10 data, you should start the IPython notebook server from the
`assignment1` directory, with the `jupyter notebook` command. (See the [Google Cloud Tutorial](http://cs231n.github.io/gce-tutorial/) for any additional steps you may need to do for setting this up, if you are working remotely)

If you are unfamiliar with IPython, you can also refer to our
[IPython tutorial](/ipython-tutorial).

### Some Notes
**NOTE 1:** This year, the `assignment2` code has been tested to be compatible with python version 3.5 (it may work with other versions of `3.x`, but we won't be officially supporting them). You will need to make sure that during your `virtualenv` setup that the correct version of `python` is used. If you use our Google Cloud virtual environment, you'll be good to go. You can confirm your python version by (1) activating your virtualenv and (2) running `which python`.

**NOTE 2:** If you are working in a virtual environment on OSX, you may *potentially* encounter
errors with matplotlib due to the [issues described here](http://matplotlib.org/faq/virtualenv_faq.html). In our testing, it seems that this issue is no longer present with the most recent version of matplotlib, but if you do end up running into this issue you may have to use the `start_ipython_osx.sh` script from the `assignment1` directory (instead of `jupyter notebook` above) to launch your IPython notebook server. Note that you may have to modify some variables within the script to match your version of python/installation directory. The script assumes that your virtual environment is named `.env`.

### Submitting your work:
Whether you work on the assignment locally or using Google Cloud, once you are done
working run the `collectSubmission.sh` script; this will produce a file called
`assignment2.zip`. Please submit this file on [Canvas](https://canvas.stanford.edu/courses/66461/).
