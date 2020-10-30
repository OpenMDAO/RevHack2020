# Using Google Colaboratory

## Overview

Colab is a service that let's you run and share Jupyter notebooks.

A notebook allows you to create and share documents that contain live Python code, equations, visualizations and narrative text. There are a variety of ways notebooks can be viewed, but Colab provides free virtual machine (VM) resources to allow you and others to run live through a web interface. This makes it easy to collaborate quickly, without having to setup and maintain your own service.

## Connecting to the Service

Colab can be found at https://colab.research.google.com/. You'll need a Google ID to get started. When connecting to that URL, you're prompted to either create a new notebook or select a previously opened one. If you're not signed in with your ID, you'll see an [intro](https://colab.research.google.com/notebooks/intro.ipynb) for Colab instead, which you should probably try if you're new to the service.

## Sharing Notebooks

Notebook files (`.ipynb` extension)  can be directly accessed through Google Drive or GitHub. When you share with someone from this service, the notebook and its contents (code, text, comments, runtime output) are shared, but the VM that it runs/ran in are **not** because each user runs in their own VM.

This means that none of the files that you added or configuration changes you made to the VM itself will be there yet for other collaborators. They'll need to run code cells to install software packages and perform other setup operations when they first connect to the Colab, so be sure to leave those necessary config steps in your notebook.

### Sharing/Saving from Colab

In the upper-right corner of the session is a Share button. It lets you create a link to share with either other Google IDs that you specify, or open it up publicly so that anyone with the link can view it. When you select individual users, they'll receive an email notifying them.

Another option is to save the .ipynb file to your Google Drive, a GitHub repo, or download it to your own system.

If you choose GitHub, you'll need to provide your GitHub login information so that Colab can get access. Colab can optionally add a text cell at the top of the notebook containing a badge which is a link to run the notebook in Colab. If you choose not to add the badge at this time, it can be manually added later with this text:

`[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)`

Replace the second URL with the link to your `.ipynb` file. The format is:

`https://colab.research.google.com/github/`*GitHub user or organization name*`/`*GitHub repo name*`/` *path to notebook file in repo*

In other words, the same as the normal URL for the notebook in your repo, but replace `https://github.com` with `https://colab.research.google.com/github`



