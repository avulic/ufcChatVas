from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template
import time
from  model import *
import pandas as pd

class PredictorAgent(Agent):    
    modelTrained = False
    file_name_fighters = './fighters.csv'
    file_name = './data_edited.csv'
    #mreza = None

    async def send_message(self, to, msg, _metadata="", _ontology=""):
        msg = Message(
            to=to,
            body=msg,
            metadata={
                "ontology": _ontology,
                "content" : _metadata,
                "languge": "english"
            })
        await self.fsm.send(msg)
        print("Posiljatelj: Poruka je poslana!" + to)

    class PredictorBehaviour(FSMBehaviour):
        async def on_start(self):
            [trainData, testData ] = LoadData(file_name)
            trainData = trainData
            input_size = trainData.columns()
            output_size = 2
            self.agent.mreza = Network(input_size, output_size)
            self.agent.fighters = pd.read_csv("fighters.csv")
            self.agent.fighters_mean = pd.read_csv("fighters_means.csv")

        async def on_end(self):
            pass

    class Predvidi(State):
        async def run(self):
            #provjeri model
            if not self.agent.modelTrained:
                self.set_next_state("Treniraj")
            else:
                try:
                    model_path = "./models"
                    if self.agent.fsm.mreza != None:
                        LoadModel(self.agent.fsm.mreza, model_path)

                
                    input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
                    predicted_label = predict(self.agent.fsm.mreza, example_input)
                    print("Predicted Label:", predicted_label.item())
                    
                    # Send the prediction result back to the chat bot agent
                    reply = Message(to="chatbot_agent@localhost")
                    reply.body = predicted_label.item()
                    await self.send(reply)
                    self.set_next_state("Primi")
                except FileNotFoundError:
                    print("Error loading model")
    
    class Primi(State):
        async def run(self):
            self.set_next_state("Primi")
            msg = await self.receive(timeout=5)
            print("Slušam poruke")
            try:
                if (msg is not None) and (msg.body is not None):
                    #izvudi konteks
                    self.agent.fighter_a = msg.metadata["content"]["fighterA"]
                    self.agent.fighter_b = msg.metadata["content"]["fighterB"]
                    #provjeri dali borci postoje i dali su u istim kategorijamaa
                    if (self.agent.fighter_a in self.agent.fighters) and (self.agent.fighter_b in self.agent.fighters):
                        #prebaci se us stanje predviđanja
                        self.set_next_state("Predvidi")
                    else:
                        sender = msg.metadata["user"]
                        await self.agent.send_message(sender, "Fighteras are not in UFC", "", "predvidi")
            except TypeError: 
                print("Error")
                self.set_next_state("Primi")

    class Odgovori(State):
        pass

    class Treniraj(State):
        async def run(self):
            modelTrained = train_model(self.agent.fsm.mreza, self.agent.fsm.trainData)
            if not modelTrained:
                modelTrained = train_model(self.agent.fsm.mreza, self.agent.fsm.trainData)
            self.set_next_state("Predvidi")

    class Dohvati_podatke(State):
        pass

    async def setup(self):
        chat_behaviour = self.PredictorBehaviour()
        self.fsm = chat_behaviour

        chat_behaviour.add_state(name="Treniraj", state=self.Treniraj(), initial=True)
        chat_behaviour.add_state(name="Primi", state=self.Primi())
        chat_behaviour.add_state(name="Predvidi", state=self.Predvidi())

        chat_behaviour.add_transition(source="Primi", dest="Primi")
        chat_behaviour.add_transition(source="Primi", dest="Treniraj")
        chat_behaviour.add_transition(source="Primi", dest="Predvidi")
        chat_behaviour.add_transition(source="Treniraj", dest="Primi")
        chat_behaviour.add_transition(source="Predvidi", dest="Primi")

        metadata = Template(metadata={"ontology": "predvidi"})
        self.add_behaviour(chat_behaviour, metadata)

chatbot_agent = PredictorAgent("ufcbrain@xmpp.jp", "12345678")
future = chatbot_agent.start()
future.result()

while chatbot_agent.is_alive():
    try:    
        time.sleep(1)
    except KeyboardInterrupt:
        print("Zaustavljam agenta...")
        chatbot_agent.stop()
        quit_spade()

