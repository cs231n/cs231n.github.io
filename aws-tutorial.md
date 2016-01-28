---
layout: page
title: Terminal.com Tutorial
permalink: /aws-tutorial/
---
For GPU instances, we also have an Amazon Machine Image (AMI) that you can use
to launch GPU instances on Amazon EC2. This tutorial goes through how to set up
your own EC2 instance with the provided AMI.

TL;DR for the AWS-savvy: Our image is `cs231n_caffe_torch7_keras_lasagne_v2`,
    AMI ID: `ami-125b2c72` in the us-west-1 region. Use a `g2.2xlarge` instance.
    Caffe, Torch7, Theano, Keras and Lasagne are pre-installed. Python bindings
    of caffe are available. It has CUDA v7.5 and CuDNN v3.

First, if you don't have an AWS account already, create one by going to the [AWS
homepage](http://aws.amazon.com/), and clicking on the yellow "Sign In to the
Console" button. It will direct you to a signup page which looks like the
following.

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-signup.png'>
</div>

Select the "I am a new user" checkbox, click the "Sign in using our secure
server" button, and follow the subsequent pages to provide the required details.
They will ask for a credit card information, and also a phone verification, so
have your phone and credit card ready.

Once you have signed up, go back to the [AWS homepage](http://aws.amazon.com),
click on "Sign In to the Console", and this time sign in using your username and
password.

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-signin.png'>
</div>

Once you have signed in, you will be greeted by a page like this:

<div class='fig figcenter fighighlight'>
  <img src='/assets/aws-homepage.png'>
</div>

Make sure that the region information on the top right is set to N. California
(For the AWS savvy, our AMI is in the `us-west-1` region). If it is not, change
it to N. California by selecting from the dropdown menu there.

(Note that the subsequent steps requires your account to be "Verified" by
 Amazon. This may take up to 2 hrs, and you may not be able to launch instances
 until your account verification is complete.)

Next, click on the EC2 link (first link under the Compute category). You will go
to a dashboard page like this:

<div class='fig figcenter fighighlight'>
  <img src='/assets/ec2-dashboard.png'>
</div>

Click the blue "Launch Instance" button, and you will be redirected to a page
like the following:

<div class='fig figcenter fighighlight'>
  <img src='/assets/ami-selection.png'>
</div>

Click on the "Community AMIs" link on the left sidebar, and search for "cs231n"
in the search box. You should be able to see the AMI
`cs231n_caffe_torch7_keras_lasagne_v2` (AMI ID: `ami-125b2c72`). Select that
AMI, and continue to the next step to choose your instance type.

<div class='fig figcenter fighighlight'>
  <img src='/assets/community-AMIs.png'>
</div>

Choose the instance type `g2.2xlarge`, and click on "Review and Launch".

<div class='fig figcenter fighighlight'>
  <img src='/assets/instance-selection.png'>
</div>

In the next page, click on Launch.

<div class='fig figcenter fighighlight'>
  <img src='/assets/launch-screen.png'>
</div>

You will be then prompted to create or use an existing key-pair. If you already
use AWS and have a key-pair, you can use that, or alternately you can create a
new one by choosing "Create a new key pair" from the drop-down menu and giving
it some name of your choice. You should then download the key pair, and keep it
somewhere that you won't accidentally delete. Remember that there is **NO WAY**
to get to your instance if you lose your key.

<div class='fig figcenter fighighlight'>
  <img src='/assets/key-pair.png'>
</div>

<div class='fig figcenter fighighlight'>
  <img src='/assets/key-pair-create.png'>
</div>

After this is done, click on "Launch Instances", and you should see a screen
showing that your instances are launching:

<div class='fig figcenter fighighlight'>
  <img src='/assets/launching-screen.png'>
</div>

Click on "View Instances" to see 
