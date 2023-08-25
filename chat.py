import spade
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State
from spade.message import Message
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response 
import time, re, json, os
import xml.etree.ElementTree as ET
import asyncio
import uuid


#ufcchat@xmpp.jp
#12345678
#ufcbrain@xmpp.jp
#12345678
class ChatBotAgent(Agent):
    database_file = 'ChatDB.sqlite3'     
    database_intent = 'IntentDB.sqlite3'           
    brainAddress = "ufcbrain@xmpp.jp"
    
    pending_requests = {} 

    async def send_message(self, to, msg, _metadata={}, _ontology=""):
        msg = Message(
            to=to,
            body=msg,
            metadata={
                "ontology": _ontology,
                "content" : json.dumps(_metadata),
                "languge": "english"
            })
        await self.fsm.send(msg)
        print("Posiljatelj: Poruka je poslana!" + to)

    
    def check_files_exist(self):
        if os.path.exists('DB/'+self.database_file) and os.path.exists('DB/'+self.database_intent):
            return True
        else:
            return False

    def saveRequest(self, sender, fighterA, fighterB):
        request_id = str(uuid.uuid4())  # Unique identifier for the request

        content = { 
            "request_id":request_id,
            "fighterA" : fighterA.lower(), 
            "fighterB": fighterB.lower(),
        }

        self.pending_requests[request_id] = {
            "user_message": content,
            "sender": sender ,
        }
        
        return self.pending_requests.get(request_id, None)["user_message"]
    
    class ChatBehaviour(FSMBehaviour):
        async def on_start(self):
            self.chatbot = ChatBot("UFC Chatbot",
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                logic_adapter=['chatterbot.logic.LogicAdapter','chatterbot.logic.BestMatch'],
                database_uri=f'sqlite:///DB/{self.agent.database_file}')

            self.chatbot_names = ChatBot('NameClassifier',
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                logic_adapter=['chatterbot.logic.LogicAdapter'],
                database_uri=f'sqlite:///DB/{self.agent.database_file}')

            self.chatbot_intent = ChatBot('intentClassifier',
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                logic_adapter=['chatterbot.logic.LogicAdapter'],
                database_uri=f'sqlite:///DB/{self.agent.database_intent}')

            
            trainer = ChatterBotCorpusTrainer(self.chatbot)
            trainer.train("chatterbot.corpus.english.conversations", "./DB/ufc.yml")
            trainer_names = ChatterBotCorpusTrainer(self.chatbot_names)
            trainer_names.train("./DB/names.yml")
            trainer_intent = ChatterBotCorpusTrainer(self.chatbot_intent)
            trainer_intent.train("./DB/ufc_intent.yml")

    class Primi(State):
        async def run(self):
            self.set_next_state("Primi")
            msg = await self.receive(timeout=5)
            print("Slušam poruke ")
            try:
                if (msg is not None) and (msg.body is not None):
                    print("Poruka primljena: " + msg.body)

                    if (len(msg.metadata) != 0) and (msg.metadata['ontology'] == "odgovor"):
                        request_id = json.loads(msg.metadata['content'])["request_id"]
                        sender = self.agent.pending_requests.pop(request_id, None)["sender"]
                        message = msg.body

                        await self.agent.send_message(sender, message)
                    else:
                        self.agent.fsm.msg = msg
                        intent = self.agent.fsm.chatbot_intent.get_response(msg.body)
                        if intent.text == "fights.outcome":
                            self.set_next_state("Predvidi")
                        else:
                            response = self.agent.fsm.chatbot.get_response(msg.body)
                            sender = msg._sender.localpart + "@" + msg._sender.domain
                            await self.agent.send_message(sender, response.text)
            except Exception as ex:
                print("Error:" + str(ex))

    class Odgovori(State):
        async def run(self):
            pass
            # msg = await self.receive(timeout=10)
            # if msg:
            #     response = self.chatbot.get_response(msg.body)
            #     if "predict" in response.text.lower() and "vs" in response.text.lower():
            #         fighters = response.text.lower().split("vs")
            #         fighter_a = fighters[0].strip()
            #         fighter_b = fighters[1].strip()

            #         self.agent.fighter_a = fighter_a
            #         self.agent.fighter_b = fighter_b
            #         self.set_next_state("Predvidi")
            #     else:
            #         await self.agent.send_message(msg.sender, response.text, "", "")


    class Predvidi(State):
        async def run(self): 
            msg = self.agent.fsm.msg
            # Check and extract the words before and after "vs"
            fighters = re.search(r"(\w+)\s+vs\s+(\w+)", msg.body)
            sender =  msg.sender[0]+ "@" + msg.sender[1]  
            if fighters is not None:
                fighter_a = fighters.group(1)
                fighter_b = fighters.group(2)
                reqContent = self.agent.saveRequest(sender, fighter_a, fighter_b)

                await self.agent.send_message(self.agent.brainAddress, "Predict fight", reqContent, "predvidi")
            else:
                await self.agent.send_message(sender, "Format: Predict Conor vs Mark")
            
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




async def main():
    print("Pokrecem")
    chatbot_agent = ChatBotAgent("ufcchat@xmpp.jp", "12345678")
    await chatbot_agent.start()
    

    while not chatbot_agent.fsm.is_killed():
        try:    
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Zaustavljam agenta")
            
            assert chatbot_agent.fsm.exit_code == 10
            await chatbot_agent.stop()

if __name__ == "__main__":
    spade.run(main())



