from toxiclib import ToxicBot
import argparse
import os

if __name__=="__main__":
	data_dir = 'data'
	if 'DATA_DIR' in os.environ:
		data_dir = os.environ['DATA_DIR']
	key = ''
	if 'API_KEY' in os.environ:
		key = os.environ['API_KEY']
	tm_url = 'localhost'
	if 'TEXT_MODEL_URL' in os.environ:
		tm_url = os.environ['TEXT_MODEL_URL']
	
	parser = argparse.ArgumentParser(description='Toxic bot runner')
	parser.add_argument('-k', '--key', help='API Key', default=key)
	parser.add_argument('-d', '--data', help='Data Directory', default=data_dir)
	args = parser.parse_args()
	bot = ToxicBot(args.data, tm_url)
	bot.run(args.key)
