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

This tutorial goes through how to set up your own Google Compute Engine (GCE) instance to work on the assignments. Each student will have $100 in credit throughout the quarter. When you sign up for the first time, you also receive $300 credits from Google by default. Please try to use the resources judiciously. But if $100 ends up not being enough, we will try to adjust this number as the quarter goes on. **Note: for assignment 1, we are only supporting python version 2.7 (the default installation from the script) and 3.5.3**.

First, if you don't have a Google Cloud account already, create one by going to the [Google Cloud homepage](https://cloud.google.com/?utm_source=google&utm_medium=cpc&utm_campaign=2015-q2-cloud-na-gcp-skws-freetrial-en&gclid=CP2e4PPpiNMCFU9bfgodGHsA1A "Title") and clicking on **Compute**. When you get to the next page, click on the blue **TRY IT FREE** button. If you are not logged into gmail, you will see a page that looks like the one below. Sign into your gmail account or create a new one if you do not already have an account. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-launching-screen.png'>
</div>

If you already have a gmail account, it will direct you to a signup page which looks like the following.
<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-for-free.png'>
</div>

Click the appropriate **yes** or **no** button for the first option, and check **yes** for the latter two options after you have read the required agreements. Press the blue **Agree and continue** button to continue to the next page to enter the requested information (your name, billing address and credit card information). Once you have entered the required information, press the blue **Start my free trial** button. You will be greeted by a page like this: 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-dashboard-screen.png'>
</div>

To change the name of your project, click on **Manage project settings** on the **Project info** button and save your changes. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-dashboard-screen.png'>
</div>

## Launch a Virtual Instance ##
To launch a virtual instance, go to the **Compute Engine** menu on the left column of your dashboard and click on **VM instances**.  Then click on the blue **CREATE** button on the next page. This will take you to a page that looks like the screenshot below. **(NOTE: Please carefully read the instructions in addition to looking at the screenshots. The instructions tell you exactly what values to fill in).**

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-create-instance-screen.png'>
</div>

Make sure that the Zone is set to be **us-west1-b** (especially for assignments where you need to use GPU instances). Under **Machine type** pick the **8 vCPUs** option. Click on the **customize** button under **Machine type** and make sure that the number of cores is set to 8 and the number of GPUs is set to **None** (we will not be using GPUs in assignment 1 and this tutorial will be updated with instructions for GPU usage). Click on the **Change** button under **Boot disk**, choose **OS images**, check **Ubuntu 16.04 LTS** and click on the blue **select** button. Check **Allow HTTP traffic** and **Allow HTTPS traffic**. Click on **disk** and then **Disks** and uncheck **Delete boot disk when instance is deleted** (Note that the "Disks" option may be hiding under an expandable URL at the bottom of that webform). Click on the blue **Create** button at the bottom of the page. You should have now successfully created a Google Compute Instance, it might take a few minutes to start running. Your screen should look something like the one below. When you want to stop running the instance, click on the blue stop button above. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-instance-started.png'>
</div>

Take note of your \<YOUR-INSTANCE-NAME\>, in this case, my instance name is instance-2. 

## Connect to Your Virtual Instance and Download the Assignment ##
Now that you have created your virtual GCE, you want to be able to connect to it from your computer. The rest of this tutorial goes over how to do that using the command line. First, download the Google Cloud SDK that is appropriate for your platform from [here](https://cloud.google.com/sdk/docs/ "Title") and follow their installation instructions. **NOTE: this tutorial assumes that you have performed step #4 on the website which they list as optional**. When prompted, make sure you select us-west1-b as the time zone. The easiest way to connect is using the gcloud compute command below. The tool takes care of authentication for you. On OS X, run:

```
./<DIRECTORY-WHERE-GOOGLE-CLOUD-IS-INSTALLED>/bin/gcloud compute ssh --zone=us-west1-b <YOUR-INSTANCE-NAME>
```

See [this page](https://cloud.google.com/compute/docs/instances/connecting-to-instance "Title") for more detailed instructions. You are now ready to work on the assignments on Google Cloud. 

Run the following command to download the current assignment onto your GCE:

```
wget http://cs231n.stanford.edu/assignments/2017/spring1617_assignment1.zip 
```

Then run:

```
sudo apt-get install unzip
```

and 

```
unzip spring1617_assignment1.zip
```

to get the contents. You should now see a folder titled assignment**X**.  To install the necessary dependencies for assignment 1 (**NOTE:** you only need to do this for assignment 1), cd into the assignment directory and run the provided shell script: **(Note: you will need to hit the [*enter*] key at all the "[Y/n]" prompts)**

```
cd assignment1 
./setup_googlecloud.sh
```

You will be prompted to enter Y/N at various times during the download. Press enter for every prompt. You should now have all the software you need for assignment**X**. If you had no errors, you can proceed to work with your virtualenv as normal.

I.e. run 

```
source .env/bin/activate
```

in your assignment directory to load the venv, and run 

```
deactivate
```
to exit the venv. See assignment handout for details.

**NOTE**: The instructions above will run everything needed using Python 2.7. If you would like to use Python 3.5 instead, edit setup_googlecloud.sh to replace the line 

```
virtualenv .env 
```

with 

```
virtualenv -p python3 .env
```

before running 

```
./setup_googlecloud.sh
```

## Using Jupyter Notebook with Google Compute Engine ##
Many of the assignments will involve using Jupyter Notebook. Below, we discuss how to run Jupyter Notebook from your GCE instance and use it on your local browser. 

### Getting a Static IP Address ###
Change the Extenal IP address of your GCE instance to be static (see screenshot below). 
<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-external-ip.png'>
</div>

To Do this, click on the 3 line icon next to the **Google Cloud Platform** button on the top left corner of your screen, go to **Networking** and **External IP addresses** (see screenshot below).

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

### Adding a Firewall rule ###
One last thing you have to do is adding a new firewall rule allowing TCP acess to a particular \<PORT-NUMBER\>. I usually use 7000 or 8000 for \<PORT-NUMBER\>. Click on the 3 line icon at the top of the page next to **Google Cloud Platform**. On the menu that pops up on the left column, go to **Networking** and **Firewall rules** (see the screenshot below). 

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-firewall-rule.png'>
</div>

Click on the blue **CREATE FIREWALL RULE** button. Enter whatever name you want: I used assignment1-rules. Enter 0.0.0.0/0 for **Source IP ranges** and tcp:\<PORT-NUMBER\> for **Allowed protocols and ports** where \<PORT-NUMBER\> is the number you used above. Click on the blue **Create** button. See the screen shot below.

<div class='fig figcenter fighighlight'>
  <img src='/assets/cloud-networking-firewall-rule-create.png'>
</div>

**NOTE:** Some people are seeing a different screen where instead of **Allowed protocols and ports** there is a field titled **Specified protocols and ports**. You should enter tcp:\<PORT-NUMBER\> for this field if this is the page you see. Also, if you see a field titled **Targets** select **All instances in the network**.

### Configuring Jupyter Notebook ###
The following instructions are excerpts from [this page](https://haroldsoh.com/2016/04/28/set-up-anaconda-ipython-tensorflow-julia-on-a-google-compute-engine-vm/ "Title") that has more detailed instructions.

On your GCE instance check where the Jupyter configuration file is located:

```
ls ~/.jupyter/jupyter_notebook_config.py
```
Mine was in /home/timnitgebru/.jupyter/jupyter_notebook_config.py

If it doesn’t exist, create one:

```
# Remember to activate your virtualenv ('source .env/bin/activate') so you can actually run jupyter :)
jupyter notebook --generate-config
```

Using your favorite editor (vim, emacs etc...) add the following lines to the config file, (e.g.: /home/timnitgebru/.jupyter/jupyter_notebook_config.py):

```
c = get_config()

c.NotebookApp.ip = '*'

c.NotebookApp.open_browser = False

c.NotebookApp.port = <PORT-NUMBER>
```

Where \<PORT-NUMBER\> is the same number you used in the prior section. Save your changes and close the file. 

### Launching and connecting to Jupyter Notebook ###
The instructions below assume that you have SSH'd into your GCE instance using the prior instructions, have already downloaded and unzipped the current assignment folder into assignment**X** (where X is the assignment number), and have successfully configured Jupyter Notebook.


If you are not already in the assignment directory, cd into it by running the following command:

```
cd assignment1 
```
If you haven't already done so, activate your virtualenv by running:

```
source .env/bin/activate
```

Launch Jupyter notebook using:

```
jupyter-notebook --no-browser --port=<PORT-NUMBER> 
```

Where \<PORT-NUMBER\> is what you wrote in the prior section.

On your local browser, if you go to http://\<YOUR-EXTERNAL-IP-ADDRESS>:\<PORT-NUMBER\>, you should see something like the screen below. My value for \<YOUR-EXTERNAL-IP-ADDRESS\> was 104.196.224.11 as mentioned above. You should now be able to start working on your assignments. 

<div class='fig figcenter fighighlight'>
  <img src='/assets/jupyter-screen.png'>
</div>

## Submission: Transferring Files From Your Instance To Your Computer ##
Once you are done with your assignments, run the submission script in your assignment folder. For assignment1, this will create a zip file called `assignment1.zip` containing the files you need to upload to Canvas. If you're not in the assignment1 directory already, CD into it by running

```
cd assignment1
```

install **zip** by running
```
sudo apt-get install zip
```

and then run 

```
bash collectSubmission.sh 
```

to create the zip file that you need to upload to canvas. Then copy the file to your local computer using the gcloud compute copy-file command as shown below. **NOTE: run this command on your local computer**:

```
gcloud compute copy-files [INSTANCE_NAME]:[REMOTE_FILE_PATH] [LOCAL_FILE_PATH]
```

For example, to copy my files to my desktop I ran:

```
gcloud compute copy-files instance-2:~/assignment1/assignment1.zip ~/Desktop
```
Another (perhaps easier) option proposed by a student is to directly download the zip file from Jupyter. After running the submission script and creating assignment1.zip, you can download that file directly from Jupyter. To do this, go to Jupyter Notebook and click on the zip file (in this case assignment1.zip). The file will be downloaded to your local computer. 

Finally, remember to upload the zip file containing your submission to [***Canvas***](https://canvas.stanford.edu/courses/66461). (You can unzip the file locally if you want to double check your ipython notebooks and other code files are correctly inside).

You can refer to [this page](https://cloud.google.com/compute/docs/instances/transfer-files "Title") for more details on transferring files to/from Google Cloud.

# BIG REMINDER: Make sure you stop your instances! #
Don't forget to stop your instance when you are done (by clicking on the stop button at the top of the page showing your instances). You can restart your instance and the downloaded software will still be available. 
