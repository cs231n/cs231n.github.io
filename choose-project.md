---
layout: page
title: Taking a Course Project to Publication
permalink: /choose-project/
---

*This tutorial was originally contributed by [Leila Abdelrahman](http://leilaabdel.com/), [Amil Khanzada](https://www.linkedin.com/in/amilkhanzada), [Cong Kevin Chen](https://www.linkedin.com/in/cong-kevin-chen-11544186/), and [Tom Jin](https://www.linkedin.com/in/tomjinvancouver/) with oversight and guidance from [Professor Fei-Fei Li](https://profiles.stanford.edu/fei-fei-li) and [Professor Ranjay Krishna](http://www.ranjaykrishna.com/).*

Taking a course project to publication is a challenging but rewarding endeavor. Starting in CS231n in Spring 2020, our team spent hundreds of hours over seven months to publish our project at the [ACM 2020 International Conference on Multimodal Interaction](http://icmi.acm.org/2020/)! We aim to share tips on creating a great project and how you can take it to the next level through publication.

<div class='fig figcenter fighighlight'>
<a href="https://github.com/fusical/emotiw">
<img alt="W3Schools" src="/assets/student-post-files/fusical.gif"> </a>
</div>

Here is a link to [our paper 📄](https://dl.acm.org/doi/abs/10.1145/3382507.3417966) and [corresponding pdf](https://github.com/fusical/emotiw/blob/master/acm_fusical_paper.pdf) for the 2020 ACM ICMI!
<br>

## Table of Contents 📖

- [CS231n: Taking a Course Project to Publication](#cs231n-taking-a-course-project-to-publication)
  - [Picking your Team 🤝](#picking-your-team-)
  - [Choosing your Project 🎛](#choosing-your-project-)
  - [Building the Project 👷🏽](#building-the-project-)
  - [Showcasing your Work 📽](#showcasing-your-work-)
  - [Refining for Publication 📄](#refining-for-publication-)
  - [Maximizing your Impact 🙌](#maximizing-your-impact-)
  - [Conclusion](#conclusion)

## Picking your Team 🤝

Although working alone may seem faster and more comfortable, it is generally not recommended. One of the primary advantages of taking a course is building a "dense network" with classmates through spending hours together working on the project. Additionally, your talented classmates may actually be teachers in disguise! If done correctly, working with a team is a lot more fun and lends for greater creativity and impact through differing opinions.

Investing good time to choose your teammates is CRITICAL, as they will shape your thinking, dictate your sanity, join your lifelong network, and ultimately determine the success of your project. Start your search as early as possible (e.g., before the course starts) so that you have enough time to consider different ideas and hit the ground running.

Have a sense of what you want to do and who you want to work with. Post your interest on Piazza and evaluate others. Join a Slack channel for your area of interest (e.g., #bio_healthcare). Interview folks through project brainstorming video calls. Consider their experience in machine learning, motivation, and personality. Specifically, when evaluating your potential team members' deep learning background, consider factors such as prior work experience, projects/publications, and AI courses in topics related to your intended project. Ask questions such as what kind of networks they would use to tackle your intended project, how they would go about finding data sources, and what tooling they are comfortable with (e.g., Google Colab or AWS). A good list of deep learning interview questions can be found on [Workera.ai](https://workera.ai/resources/deep-learning-algorithms-interview/).

Finally, be flexible and pick your group. Always keep in mind that plans can change, and some students do end up dropping the course. Be open in your communication and watch out for unexpected changes in your group before the course drop deadline.

### Picking your Mentors
Do note that mentorship and support are invaluable in ensuring your success. Find the TA, professor, and/or industry mentors best suited to guide you and meet with them often. Course staff and mentors are generally interested in helping YOU succeed and may even write you a recommendation letter in the future!

<div class='fig figcenter fighighlight'>
  <img src='/assets/student-post-files/choose-project-team.png' width=500px>
</div>

### What We Did
We looked at the profiles of all the TAs and went to office hours. We were lucky to find Christina Yuan, who had done similar research work and soon became a kind mentor for us. Professor Ranjay Krishna was also very kind to guide us in taking our project to publication.

## Choosing your Project 🎛

What you do is very important, as it determines your enjoyment and builds your foundation for future research.

### Sample Categories of CS231n Projects

Past projects can be found here on the [the course website](http://cs231n.stanford.edu/2020/project.html) and include categories such as:

1. Image Classification
2. Image Segmentation
3. Video Processing
4. 3D Deep Learning
5. Sentiment Analysis
6. Object Detection
7. Data Augmentation
8. Audio Analysis with CNNs and Spectrogram Images

### Consider [SMART Goals](https://en.wikipedia.org/wiki/SMART_criteria)

#### Specific

What question are you exactly trying to answer, or what concept do you want to address in-depth? Specific projects are suitable for focus, but make sure it has some generalizability.

*Good Example:*
Generate Deep Fakes

*Bad Example:*
Generating Deep Fakes on lung CT scans to improve medical image segmentation.


#### Measurable
  Make sure there are performance metrics, qualitative assessments, and other benchmarks you can use to measure your work's success. Examples include accuracy on a test dataset and performance relative to a human.

*Bad Example:*
Create high-fidelity image reconstructions.

*Good Example:*
Create high-fidelity image reconstructions with low mean-squared error difference compared to original images.

#### Attainable
Do you have the resources, tools, and background experience to accomplish the project's tasks? Choosing grandiose projects beyond reach can only bring disappointment, be realistic. Ask yourself if you have the computing power (GPUs, disk space), the data (is it open-access, private, do you have to curate it yourself), and the bandwidth (are you juggling a busy semester/quarter).

If you are currently doing CV related research, you can incorporate that into your project, and your supervisor may be willing to lend you data or compute resources. If not, there is usually a short list of faculty-sponsored projects looking for students.

*Bad Example:*
Detection of cancer from facial videos.<br> This is not attainable because there is unlikely to be a large enough dataset that is open-access.

*Good Example:*
Projects that can leverage open-source repos and APIs or work with pre-trained or fine-tuned Deep Learning models.

#### Relevant
CS231n requires you to process some form of 2D pixelated data with a CNN. <br> <br> Discussing your idea with the TAs is good to know how relevant the project is to the class. Make sure the project also aligns with your values and interests. Think long term, and ask yourself five years after the project ends if it enriched your interests and career direction.

*Bad Example:*
Chirp classification with principal component analysis.

*Good Example:*
Object recognition using CNNs. We incorporated feature detection tasks like scene analysis and image captioning for emotion recognition in videos in our project.

#### Timely
CS231n only lasts for 8-10 weeks. Make sure you can deliver your objectives along the class's timeline. Roadmap your project to see what needs to get done--and when! This is perhaps the most critical. <br> <br> Allow yourself 2-3 weeks to find a team and formulate a proposal, 3-4 to collect/preprocess data and train your first model against an established baseline (milestone), and the rest to run additional models, tune hyperparameters, and format your report. <br> <br> It's helpful to log everything in Overleaf and repurpose your milestone for your final submission so that everything doesn't catch up to you in the last few days.

*Bad Example:*
A project that requires two years to collect, clean, and organize raw data.

*Good Example:*
A project where you can work with existing datasets or spend a few weeks collecting raw data. For our EmotiW project, an existing dataset was provided to us, so we were able to spend more time on algorithms.

### Advice from Professor Fei-Fei Li

1. **Maximize learning:** Find a project that can enable your maximal learning.
2. **Feasibility:** Make sure it is executable within the short amount of time in this quarter, especially making sure that data, evaluation metric, and compute resources are adequate for what you want to do.
3. **Think long term:** Executing a good project regardless of the topic would benefit your job application. Companies look for people with a solid technical background and the ability to execute and work as a team. I don't think they will be very narrow-minded on the specific project topics.

## Building the Project 👷🏽

When working on the project, consider these tips to maximize the potential for impact.

### Work in the Cloud

Store documentation, code, datasets, and model files in services such as Google Drive, Dropbox, or GitHub.

The course offers AWS and GCP credits for your projects. Consider taking advantage of these resources, especially when it comes to heavy computing.

If you plan to use Colab, consider saving your model checkpoints directly to your Stanford Google Drive folder to avoid losing files. Colab is also great to parallelize your work as you can run multiple experiments simultaneously. Be careful not to let your work session time out, or you could lose your progress!


### Work Collaboratively
Brainstorm your projects on live-editors like Google Sheets and Google Docs so others can work in the same workspace. Overleaf is excellent for polishing the final report as a team.

Set up regular weekly or semi-weekly meetings with your teammates. This establishes a cadence and helps keep everyone on track, even if no progress has been made.

### Document Everything
Write your code as if you will be showing it to someone who has no idea about your project. This is great for making your methods transparent and reproducible.

Record all experiments you run (e.g., hyperparameter tuning) even if they don't work out, as you can write about them in your report.


### Be Organized
Create clean file structures. Give folders and scripts precise names. Organization saves your team hours, if not days, in the long run.

Write your notebooks, so they are executable top-to-bottom. That way, if you suffer a catastrophic data loss or discover a mistake in your code, you can smoothly go back and replicate your entire setup.

## Showcasing your Work 📽

Communication makes the project memorable. Come up with a fun title, and make sure to present your work in multiple ways. Beyond the written report, create a slide deck, record a video, and practice talking about your work with TAs, professors, and industry experts. Articulation skills in oral and visual presentations ensure that the audience remembers your work!

Before completing the project, take the time to clean up your GitHub repository, or make one if you have not done so already. Add a strong README.md file with a clear description of your project and a "how-to" guide complete with examples. Doing so makes your project accessible and encourages others to build upon your work.

If you are submitting to a journal or conference, consider submitting a copy of your work to a preprint service like [arXiv](https://arxiv.org/). arXiv does not require peer-review and is commonly used by researchers within the CS field to disseminate their work. Publishing a preprint is a quick way to add to your list of publications on Google Scholar. You may also want to take 5 minutes to register and [ORCID](https://orcid.org/).

### Our Challenges
With four people working simultaneously, our Github repository had naturally become a little hard to navigate since everyone used a slightly different naming scheme. Multiple cloned notebooks were also leftover from other experiments. In our case, our goal had always been to submit a grand challenge paper based on our results, so organizing public-facing repository by project completion was critical.

In 2020, many conferences were entirely virtual, which made it even more important to have a polished video presentation as this would end up as part of conference proceedings.

### What We Did
We made sure our Github repository reflected our results, and we took the opportunity to embed multiple figures into the README to make it visually appealing and easily understandable.

We chose the playful title "Fusical" and created a one-figure visual that summarized our work at a high-level. This made it easy for people to reference our work during the conference, allowing us to talk about our project without confusing attendees with different slides.

## Refining for Publication 📄

When refining for publication, ask yourself: *What am I bringing to the conversations that is new?* This can come from novelty or exceeding state-of-the-art methods. Reinforcing this question throughout refinement is critical for successfully publishing.

The next step is to find the right journal or conference to publish to. Unlike most science fields, CS and AI tend to value conference submissions over journal submissions due to the publication turnaround time and higher visibility within the field. Conferences are fun and allow you to network with other like-minded individuals.

In Computer Vision, you'll find conferences of all sizes happening at different times throughout the year. Top conferences include [CVPR](http://cvpr2021.thecvf.com/), [ICCV](http://iccv2021.thecvf.com/home), and [ECCV](https://eccv2020.eu/), but due to their popularity, it might be challenging to get a successful submission. Because many students choose to do an application-based project, consider niche conferences depending on your project's topic. For instance, teams who developed a model to determine student engagement in the classroom might consider submitting to [AIED](https://aied2020.nees.com.br/). Talking to your mentors might also help narrow down the specific method or venue of publication. Keep in mind that Stanford does offer some [travel grants](https://undergrad.stanford.edu/opportunities/research/get-funded).


Iterating over the work is essential for creating publication-quality work. Improving the writing, optimizing the code, and making quality figures are vital points to consider. Reference the work of experts and reach out for advice. [Dr. Juan Carlos Niebles' publications page](http://www.niebles.net/publications/ ) is a good example.

Publications are a lengthy process, partly due to peer review. Review your submission for possible ambiguous areas or claims that are unsubstantiated. Ensure you are clear with your experiment parameters and ask someone outside your team to review the paper.

Writing for a peer-reviewed journal or conference proceedings is very different than preparing a project report, yet the project helps guide what goes into the final paper.

### Our Challenges
While we made advances during the CS231n course, we didn't believe our models were novel enough for publication just yet. Based on guidance from our mentors, we looked for opportunities to differentiate our work from others. Because our project was part of the [EmotiW grand challenge](https://sites.google.com/view/emotiw2020), one of the motivating factors was to continue exploring additional methods to improve our validation performance before the submission deadline.

### What We Did
Based on a literature search, we found common patterns in the modalities used by other groups for sentiment classification (e.g. audio, pose). As a result, we focused on exploring two novel modalities that we felt had value: image captioning (to look for certain objects or settings, such as protests, that are correlated with a specific degree of sentiment) and laughter detection (since the presence of laughter in an audio clip quite often indicates a positive sentiment). Using modified saliency maps and ablation studies, we tried to demonstrate the usefulness of all of these modalities for a more convincing argument.

Overall, we attained a test accuracy of approximately 64%. Because there was no public leaderboard, there was initial uncertainty about how we fared compared to other groups, but we were confident in our approach as it beat the baseline by a large margin. Although we ultimately ranked below the top three winning scores, we were still able to publish our work at the conference due to our novel findings, demonstrating that the test data performance is only of many criteria stressed for publication.

### Differences Between our Course Project and Publication Paper

#### Course Project
- More focus on error analysis and experiments that went wrong.
- Tendency towards experimental approaches, as the course timeframe was too short for pretraining and auxiliary data.
- Significant time spent on "invisible" setup work for the project's infrastructure (preprocessing, APIs, and overall backbone).
- Our original CS231n original project submission can be found [here](https://github.com/fusical/emotiw/blob/master/cs231n_project_report.pdf).

#### Publication Paper
- More focus on the novelty of our approach and emphasis on beating the competition baseline.
- The video presentation was longer and required more detail.
- Ran more experiments and hyperparameter tuning.
- Our official published paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3382507.3417966) and [as a pdf](https://github.com/fusical/emotiw/blob/master/acm_fusical_paper.pdf).

*If you don't get the results you were hoping for, don't be scared to submit it anyway - the worst reviewers can do is say no, and feedback from experts in the field is always valuable!*

## Maximizing your Impact 🙌
Your project has worth and impact. Presenting at global conferences (we presented at the 2020 ICMI) is a great way to share your results and findings with other academic and industry experts, but there are other options. Do you think your idea can be deployed in practice? Stanford has resources and opportunities to help students continue to pursue their ideas, including the [Startup Garage](https://www.gsb.stanford.edu/experience/learning/entrepreneurship/courses/startup-garage), a hands-on project-based course to develop and test out new business concepts, and the [Vision Lab](http://vision.stanford.edu/people.html).

If your idea is more research-focused, seek out professors who might work in the domain area and pitch them your thoughts. You might have opportunities to continue your project as an independent study project or apply your idea to more extensive and significant datasets or pivot to similar research areas.

## Conclusion

It was hard work and fun because we had a good team! We hope you all enjoy these tips on the process of taking a project from brainstorming to impactful presentations and publications. These tips are scalable, and we hope you use them in other endeavors as well!
