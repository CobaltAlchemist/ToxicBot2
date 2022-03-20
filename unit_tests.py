import unittest
from toxiclib.predictor import Predictor
from toxiclib.constants import *
from toxiclib.ddb import DiscordDatabase
from discord import Guild, User, TextChannel, Message

class TestPredictor(unittest.TestCase):
	def setUp(self):
		self.p = Predictor('model', 'cpu')

	def test_prediction(self):
		preds = self.p("Hello world")
		self.assertEqual(len(preds), len(TOXCLASSES_HUMAN))
		self.assertLessEqual(max(preds), 1.)
		self.assertGreaterEqual(min(preds), 0.)

	def test_logits(self):
		preds = self.p("Hello world", logits=True)
		self.assertEqual(len(preds), len(TOXCLASSES_HUMAN))

class TestDatabase(unittest.TestCase):
	def setUp(self):
		self.db = DiscordDatabase('test')
		self.testuser = User(state=None, data=None)
		self.testchannel = TextChannel(state=None, data=None)
		self.testguild = Guild(state=None, data=None)
		self.testmessage = Message(state=None, data=None)
		self.testuser.name = "joe"
		self.testuser.discriminator = "1234"
		self.testchannel.id = 100
		self.testchannel.guild = self.testguild
		self.testguild.id = 200
		self.testmessage.author = self.testuser
		self.testmessage.guild = self.testguild
		self.testmessage.channel = self.testchannel
		self.testmessage.content = "Hello world"
		self.testmessage.content = "Hello world"

	def tearDown(self):
		os.remove('test\\ud.csv')
		os.remove('test\\lm.csv')
		os.remove('test\\hist.csv')

	def test_add_message(self):
		self.db.add_message(self.testmessage)
		self.assertEqual(len(self.db.user_data), 1)

if __name__ == '__main__':
	unittest.main()
