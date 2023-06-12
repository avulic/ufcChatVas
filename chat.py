from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State
from spade.message import Message
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import time, re, os
import xml.etree.ElementTree as ET




#ufcchat@xmpp.jp
#12345678
#ufcbrain@xmpp.jp
#12345678
class ChatBotAgent(Agent):
    database_file = 'ChatDB.sqlite3'             
    brainAddress = "ufcbrain@xmpp.jp"

    async def send_message(self, to, msg, _metadata=""):
        msg = Message(
            to=to,
            body=msg,
            metadata={
                "ontology": "predvidi",
                "content" : _metadata,
                "languge": "english"
            })
        await self.fsm.send(msg)
        print("Posiljatelj: Poruka je poslana!" + to)

    def classify_human_name(self, response):
        bot_response = self.chatbot_names.get_response(response)

        contains_human_name = False
        words = bot_response.text.split()
        if (("yes" in words) or ("is" in words)) and (("no" not in words) and ("not" not in words)):
            contains_human_name = True
        #for word in words:
        #    if contains_human_name and word.isinit():
        #        human_name = word
        #        break

        return contains_human_name

    class ChatBehaviour(FSMBehaviour):
        async def on_start(self):
            # Create a ChatterBot instance
            self.chatbot = ChatBot("UFC Chatbot",
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                logic_adapter=['chatterbot.logic.LogicAdapter','chatterbot.logic.BestMatch'],
                database_uri=f'sqlite:///{self.agent.database_file}')
            
             # Create a new chat bot
            self.chatbot_names = ChatBot('NameClassifier',
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                logic_adapter=['chatterbot.logic.LogicAdapter'],
                database_uri=f'sqlite:///{self.agent.database_file}')
            
            
            #trainer = ChatterBotCorpusTrainer(self.chatbot)
            #trainer.train("chatterbot.corpus.english.conversations",
            #            "./ufc.yml")
            #trainer_names = ChatterBotCorpusTrainer(self.chatbot_names)
            #trainer_names.train("./names.yml")

    class Primi(State):
        async def run(self):
            msg = await self.receive(timeout=5)
            print("Slušam poruke")
            try:
                if (msg is not None) and (msg.body is not None):
                    self.agent.fsm.msg = msg
                    response = self.agent.fsm.chatbot.get_response(msg.body)
                    if ("predict" in msg.body.lower()) or ("vs" in msg.body.lower()):
                        if ("predicti" in response.text.lower()) and ("vs" in response.text.lower()) and ("yes" in response.text.lower()):
                            self.set_next_state("Predvidi")
                    else:
                        sender = msg._sender.localpart + "@" + msg._sender.domain
                        await self.agent.send_message(sender, response.text)
                else: 
                    self.set_next_state("Primi")
            except:
                print("Error")
    
    class Odgovori(State):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                response = self.chatbot.get_response(msg.body)
                if "predict" in response.text.lower() and "vs" in response.text.lower():
                    fighters = response.text.lower().split("vs")
                    fighter_a = fighters[0].strip()
                    fighter_b = fighters[1].strip()

                    to = self.agent.brainAddress
                    msg = f"Predict {fighter_a} vs {fighter_b}"
                    await self.send_message(to, msg)
                    self.agent.fighter_a = fighter_a
                    self.agent.fighter_b = fighter_b
                    self.set_next_state("Predvidi")
                else:
                    reply = Message(to=msg.sender, body=response.text)
                    await self.send(reply)

    class Predvidi(State):
        async def run(self): 
            # Extract the words before and after "vs"
            fighters = re.search(r"(\w+)\s+vs\s+(\w+)", self.agent.msg)
            fighter_a = fighters.group(1)
            fighter_b = fighters.group(2)
            print("a:"+fighter_a + " " + "b:"+fighter_b)
            
            a = self.agent.classify_human_name(fighter_a)
            b = self.agent.classify_human_name(fighter_b)
            if a == True and b == True:
                metadata= { "borac_a" : self.agent.fighter_b, "borac_b": self.agent.fighter_b }
                self.agent.send_message(metadata)
            self.set_next_state("Primi")

    async def setup(self):
        chat_behaviour = self.ChatBehaviour()
        self.fsm = chat_behaviour
        chat_behaviour.add_state(name="Primi", state=self.Primi(), initial=True)
        chat_behaviour.add_state(name="Odgovori", state=self.Odgovori())
        chat_behaviour.add_state(name="Predvidi", state=self.Predvidi())

        chat_behaviour.add_transition(source="Primi", dest="Odgovori")
        chat_behaviour.add_transition(source="Odgovori", dest="Primi")
        chat_behaviour.add_transition(source="Primi", dest="Predvidi")
        chat_behaviour.add_transition(source="Predvidi", dest="Primi")
        
        chat_behaviour.add_transition(source="Primi", dest="Primi")
        
        self.add_behaviour(chat_behaviour)

chatbot_agent = ChatBotAgent("ufcchat@xmpp.jp", "12345678")
future = chatbot_agent.start()
future.result()

while chatbot_agent.is_alive():
    try:    
        time.sleep(1)
    except KeyboardInterrupt:
        print("Zaustavljam agenta...")
        chatbot_agent.stop()
        quit_spade()

