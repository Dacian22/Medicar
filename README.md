# medicar: Intelligent Hospital Logistics
A European Team Project from Sergiu-Dacian Balint, Diana Matei, Sara Koni, Arved Schreiber and Paul König
## Ressources
* Main Overview Slidedeck: (Application-Architecture, message flow, etc.): https://1drv.ms/p/s!AmfhbH57g19vh2GM2p8LEDvWmaYj
* Onboarding Slidedeck + Funevents planning: https://1drv.ms/p/s!AmfhbH57g19vh2pFDChNDunJDYHc
* Trello Board (for task management): https://trello.com/b/3Z3z3z3z/medicar
## Quickstart
* Install the required packages: `pip install -r requirements.txt`
* Create .env file in the root directory according to the .env.example file (ask paul for credentials)
## Development Guide
### Environment
Currently, we only use python and the dependencies in the `requirements.txt` file. For the visualization, we may use 
some JS in future.
### Architecture and File Structure
The project is structured in a modular way. Each module is responsible for a specific task. The main modules for now are: 
* Order Manager
* Simulation (also includes the LLM)
* Vehicle
Each Module will be in its own directory and will have its own tests. Communication between the modules is done solely via MQTT, otherwise they are completely decoupled.
For more information on the architecture, please refer to the main overview slidedeck. Each module, but especially the `Vehicle` module, 
should be built to be converted into a single deployment unit as easily as possible. 
### MQTT Broker
We are using the free-tier MQTT Broker from HiveMQ. You can find the URL and Port inside the .env.example file. Please 
copy the `.env.example` file to your personal and secret `.env` fill and fill in the credentials provided by Paul.
For interaction with the MQTT Broker, we are using the `paho-mqtt` library. For an example on how to use it, please refer to the `Vehicle` module.

We are trying to use, if possible and reasonable, the "VDA5050" standard for MQTT message protocols in the automotive industry.
You can find it in the `Ressources/MQTT_APIS` folder as PDF and as txt. You can visualize the txt files nicely with the following
online service https://studio.asyncapi.com/ (just paste the content there)

If you want to monitor the mqtt messages in real time you can use the following open source app: https://mqttx.app/downloads

### Git "Workflow"
For now, we want to use only the main branch. This may change, but currently we want full transparency. So please
commit directly to main, even if your code isn't finished yet. If you want to try something out, please create a folder
called `Playground_<YourName>` and push it. This way we have transparency about what is currently being worked on.
We consider it as best-practise to push at least once a day, if you have some changes.
