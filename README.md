# FastAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Implementation

Objective is to create an API endpoint for "sentiment classification model", which aims to predict if a given text message is spam or ham.

## Contents

1. [Requirements](#requirements)
2. [Quickstart](#quickstart)



## Requirements

1. Install required packages (fastapi, uvicorn etc..,)

   pip install -r requirements.txt 
   
2.  Install [Docker](https://www.docker.com/community-edition#/download)
 and [Docker Compose](https://docs.docker.com/compose/install/)


## Quickstart

**Step 1: Building the API**

* Client sends a request to the uvicorn server which interacts with the API to trigger the prediction model.
* The model returns the polarity (spam or ham), and the result is shown to the user in a JSON format.

**Step 2: Deploying into Docker**

* After creating the API, we will create a Docker Image in which the app will the running.


## About files

**app.py** containing all the instructions on the server-side
**docker** containing the Dockerfile to create the container.


# Steps to Build the API

1. The API implementation in the python fileapp.py is split into the following three main sections.

2. Import all the libraries required for the API (lines 2 to 3). The step before that is to install those libraries, which can be done by running the pip install -r requirements.txt command.

 Load the serialized model and the vectors (lines 6 and 9) and instantiate the Fast API (line 12)

3. Define the endpoints of the API. In our case, it will have two routes as shown below.

**default route "/”:** which simply returns the following JSON format “message”: “Welcome to Your Sentiment Classification FastAPI” through the root() function which does not take a parameter.

**the prediction route "/predict_sentiment":** this one triggers the predict_sentiment() function, which takes as a parameter, the user’s message, and returns the JSON response at the format defined from lines 19 to 22. The sentiment polarity is the prediction of the model.

We can finally run it (the API) with the following instruction from our command line, from the location of the app.y file.

uvicorn app:app --reload

* uvicorn starts the unicorn server.
* app: corresponds to the python fileapp.pyIt is the one on the left of the “:” sign.
* app: corresponds to the object created inside of app.py with the instruction app = FastAPI(). If the name of the file was main.py the instruction would be uvicorn main:app --reload
* --reload: an option used to restart the server after any code changes. Keep in mind that this is to be used only when we are in the development stage, not during the deployment stage.

Line 1 shows that the unicorn server is running on the localhost (http://127.0.0.1) port 8000.

Accessing the default route is straightforward. Just type the following URL on our favorite browser.


The URL http://127.0.0.1:8000/docs provides a complete dashboard for interacting with our API. Below is the result.

From the previous Image, when we select the /predict_sentiment in the green box, we get the complete guide to interact with our API as shown below.

Just select the Try it out tab and provide the message you want the prediction for in the text_message zone.

## Deployment into Docker Container
Our API is ready, now it is time to deploy it into a Docker container. The idea behind containerization is that it will make our API portable and able to run uniformly and consistently across any platform (including the cloud), in a more secured way. Also, using Kubernetes can make the scaling of the API easier. But Kubernetes' part will be for another time.

Below is the content of theDockerfile for our app.

The Dockerfile file contains five main instructions as shown below.

* FROM: pulls an official python image from docker hub, then our files will be built from that image.
* WORKDIR: create /app as the working directory for the application.
* COPY: copy file(s) from the source folder to the destination folder.
* RUN: runs the requirements.txt file in order to install the project dependencies.
* CMD: create an entry point in order to finally make the image executable.


## Build the Docker Image

The following command builds the image called fastapiapp tagged with the latest version. This image is built from the previous docker file.

Running the command creates the image with the following success message

## Run the container

The image is built, all we need now is to run it into a container with the following command.

**After that, you should get a message similar to this one containing the URL to your API.**


## Docker Compose:

Now we’ll define a docker-compose that will create a service for our API. I could have just used a Dockerfile only because docker-compose is a tool for defining multi-container applications and we have one container so far.

#### Using Compose is basically a three-step process:
* Define app's environment with a Dockerfile so it can be reproduced anywhere.
* Define the services that make up an app in docker-compose.yml so they can be run together in an isolated environment.
* Run docker-compose up and compose will start and run the entire app.

#### Refer docker-compose.yml file:

We give a name to our service (service), we set the path to the Dockerfile for the build step, and we map the port 80(on the container) to the port 80 (on the host).

#### Start Docker Compose using the following command

**docker-compose up**

Docker compose is great tool to manage multiple services at a time, with few lines of YAML you can set up a whole infrastructure for your app all in one file.
