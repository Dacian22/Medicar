# InES Team Project SS24 - Intelligent Hospital Logistics
> A European Team Project from Sergiu-Dacian Balint, Diana Matei, Sara Koni, Arved Schreiber and Paul König.
## Introduction
### Background
The coronavirus pandemic has shown that exceptional medical situations, coupled with staff shortages, can push entire hospital systems to their limits. The research project ”MediCar 4.0” seeks to create an advanced transport logistics platform for self-driving vehicles on clinic premises to avoid future bottlenecks and enhance operational efficiency. However, the dynamic nature of emergencies, failures, and other unpredictable events poses challenges. Many conflicts, such as congestion and deadlocks, can only be adequately resolved with expert knowledge, causing classical optimization algorithms to fall short.
### Project Goal
This team project aims to develop robust AI-based routing strategies that integrate world-knowledge from large language models to meet the challenges of clinical processes and various supplies. In this context, we build a simulation environment of the University Hospital in Freiburg and implement routing algorithms to test them in various scenarios.
## Quickstart
* Install the required packages: `pip install -r requirements.txt`
* Create .env file in the root directory according to the .env.example file
* Start the main.py scripts in the following order: First Simulation, second Vehicle and third OrderManager
## Development Guide
### Repository Structure
The project is structured in a modular way. Each module is responsible for a specific task. The main modules: 
* `Simulation`
* `Vehicle`
* `Order Manager`

Communication between the modules is done solely via MQTT, otherwise they are completely decoupled.
### MQTT Broker
We are using the free-tier MQTT Broker from HiveMQ. You can find the URL and Port inside the .env.example file. Please 
copy the `.env.example` file to your personal and secret `.env` file and fill in the credentials provided by Paul.
For interaction with the MQTT Broker, we are using the `paho-mqtt` library. For an example on how to use it, please refer to the `Vehicle` module.

We are trying to use, if possible and reasonable, the "VDA5050" standard for MQTT message protocols in the automotive industry.
You can find it in the `Ressources/MQTT_APIS` folder as PDF and as txt. You can visualize the txt files nicely with the following
online service https://studio.asyncapi.com/ (just paste the content there).

If you want to monitor the mqtt messages in real time you can use the following open source app: https://mqttx.app/downloads

## LLM Evaluation
The following two pictures show the final evaluation results of our large language models. For further explaination of the models feel free to take a look at our presentation slides in the folder `Artifacts/Presentation-Slides`. 
### Submodels
In the following picture you can see the evaluation results of our independent submodels.

![LLM Results Submodels](docs/LLM_results_submodels.png)
### Metamodel
This picture shows the accuracies of the final metamodel.

![LLM Results Metamodel](docs/LLM_results_metamodel.png)
