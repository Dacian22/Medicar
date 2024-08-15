# InES Team Project SS24 - Intelligent Hospital Logistics
> A European Team Project of students from the Babeș-Bolyai University in Cluj-Napoca and the University of Mannheim in the spring semester 2024.

>**Participants**:
>* Sergiu-Dacian Balint
>* Diana Matei
>* Sara Koni
>* Arved Schreiber
>* Paul König
## Introduction
### Background
The coronavirus pandemic has shown that exceptional medical situations, coupled with staff shortages, can push entire hospital systems to their limits. The research project ”MediCar 4.0” seeks to create an advanced transport logistics platform for self-driving vehicles on clinic premises to avoid future bottlenecks and enhance operational efficiency. However, the dynamic nature of emergencies, failures, and other unpredictable events poses challenges. Many conflicts, such as congestion and deadlocks, can only be adequately resolved with expert knowledge, causing classical optimization algorithms to fall short.
### Project Goal
This team project aims to develop robust AI-based routing strategies that integrate world-knowledge from large language models to meet the challenges of clinical processes and various supplies. In this context, we build a simulation environment of the University Hospital in Freiburg and implement routing algorithms to test them in various scenarios.
## Quickstart
* Install the required packages: `pip install -r requirements.txt`
* Create .env file in the root directory according to the .env.example file
* Start the main.py scripts in the following order: First Simulation, second Vehicle and third OrderManager
* To generate prompts by the vehicles, activate the button under the tab "Prompts"
## Repository Structure
The project is structured in a modular way. Each module is responsible for a specific task. The main modules are: 
* `Simulation`
* `Vehicle`
* `OrderManager`

Communication between these modules is done solely via MQTT, otherwise they are completely decoupled.


The folder `Artifacts` contains pdf-files with presentation slides and information about the LLM evaluation.
`Resources` includes csv-files with the LLM outputs for evaluation purposes as well as a heuristic file of orders and the osm-file of the University Hospital of Freiburg. The folder `docs` exists to include picture in the README.
### Module `Simulation`
The module `Simulation` connects the two other main modules `OrderManager` and `Vehicle`.
The `OrderManager` sends an MQTT-message to the `Simulation` under the topic “/order”. The module `Simulation` creates the route for the order and sends it via MQQT to `Vehicle` under the topic “/vehicles” such that the order can be fulfilled. Furthermore, `Simulation` includes all the LLM models to handle the impact of events to the graph and the routing of the vehicles.
| File                      | Role                                                     |
|---------------------------|----------------------------------------------------------|
| `main.py`                 | Main script to build the graph and activate the routing  |
| `BuildGraph.py`           | Source code of graph creation                            |
| `Routing.py`              | Source code of the class Routing                         |
... LLM files

### Module `Vehicle`
| File                      | Role                                                     |
|---------------------------|----------------------------------------------------------|
| `main.py`                 |                                                          |
| `Vehicle.py`              |                                                          |

### Module `OrderManager`
| File                      | Role                                                     |
|---------------------------|----------------------------------------------------------|
| `main.py`                 |                                                          |
| `order.py`                |                                                          |
| `order_manager.py`        |                                                          |

## LLM Evaluation
The following two pictures show the final evaluation results of our large language models. For further explaination of the models feel free to take a look at our presentation slides in the folder `Artifacts/Presentation-Slides`. 
### Submodels
In the following picture you can see the evaluation results of our independent submodels.

![LLM Results Submodels](docs/LLM_results_submodels.png)
### Metamodel
This picture shows the accuracies of the final metamodel.

![LLM Results Metamodel](docs/LLM_results_metamodel.png)
