from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template
import time, json, os
from  model import *
import pandas as pd

class PredictorAgent(Agent):    
    modelTrained = False
    file_name_fighters = './fighters.csv'
    file_name = './data_edited.csv'
    mreza = None
    trainData = None
    fighters = None
    fighters_mean = None

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
            self.agent.trainData = trainData
            input_size = trainData.columns()
            output_size = 2
            self.agent.mreza = Network(input_size, output_size)
            self.agent.fighters = pd.read_csv("fighters.csv")
            self.agent.fighter_mean = pd.read_csv("fighter_means.csv")
            
        async def on_end(self):
            pass

    class Predvidi(State):
        async def run(self):
            #provjeri model
            if not self.agent.modelTrained:
                self.set_next_state("Treniraj")
            else:
                try:           
                             
                    fighter_a = self.agent.fighter_mean[self.agent.fighter_mean["Name"] == self.agent.fighter_a]
                    fighter_b = self.agent.fighter_mean[self.agent.fighter_mean["Name"] == self.agent.fighter_b]

                    fighter_a.reset_index(drop=True, inplace=True)
                    fighter_a = fighter_a.drop('Name', axis=1)
                    fighter_b.reset_index(drop=True, inplace=True)
                    fighter_b = fighter_b.drop('Name', axis=1)

                    fighter_a = fighter_a.rename(columns=lambda x: "R__" + x if ((x != 'Last_round') and (x != 'Max_round')) else x)
                    fighter_b = fighter_b.rename(columns=lambda x: "B__" + x if ((x != 'Last_round') and (x != 'Max_round')) else x)

                    all_conc = pd.concat([fighter_a, fighter_b], axis=1)               
                    input = torch.tensor(all_conc.values, dtype=torch.float32)
                    predicted_label = predict(self.agent.mreza, input)
                    print("Predicted Label:", predicted_label.item())
                    

                    self.set_next_state("Primi")
                except Exception as e:
                    print("Error:" + str(e))
    
    class Primi(State):
        async def run(self):
            self.set_next_state("Primi")
            msg = await self.receive(timeout=5)
            print("Slušam poruke")
            try:
                if (msg is not None) and (msg.body is not None):
                    #izvudi konteks
                    self.agent.fighter_a = json.loads(msg.metadata["content"])["fighterA"]
                    self.agent.fighter_b = json.loads(msg.metadata["content"])["fighterB"]
                    #provjeri dali borci postoje i dali su u istim kategorijamaa
                    self.agent.fighter_a = self.agent.fighters[self.agent.fighters['Fighters'].str.contains(self.agent.fighter_a)]['Fighters'].values[0]
                    self.agent.fighter_b = self.agent.fighters[self.agent.fighters['Fighters'].str.contains(self.agent.fighter_b)]['Fighters'].values[0]
                    if (self.agent.fighter_a != "") and (self.agent.fighter_b != ""):
                        #prebaci se us stanje predviđanja
                        self.set_next_state("Predvidi")
                    else:
                        sender = msg.metadata["user"]
                        await self.agent.send_message(sender, "Fighteras are not in UFC", "", "predvidi")
            except TypeError: 
                print("Error")
                self.set_next_state("Primi")

    class Treniraj(State):
        async def run(self):
            try:
                self.agent.modelTrained = train_model(self.agent.mreza, self.agent.trainData)
                if not self.agent.modelTrained:
                    self.agent.modelTrained = train_model(self.agent.mreza, self.agent.trainData)
                
                self.set_next_state("Primi")
            except TypeError: 
                print("Error")
                self.set_next_state("Primi")

    async def setup(self):
        chat_behaviour = self.PredictorBehaviour()
        self.fsm = chat_behaviour

        chat_behaviour.add_state(name="Treniraj", state=self.Treniraj(), initial=True)
        chat_behaviour.add_state(name="Primi", state=self.Primi())
        chat_behaviour.add_state(name="Predvidi", state=self.Predvidi())

        chat_behaviour.add_transition(source="Primi", dest="Primi")
        chat_behaviour.add_transition(source="Primi", dest="Predvidi")
        chat_behaviour.add_transition(source="Treniraj", dest="Primi")
        chat_behaviour.add_transition(source="Predvidi", dest="Treniraj")
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

