from toxiclib import ToxicBot
import argparse

if __name__=="__main__":
	bot = ToxicBot(model_dir='model')
	parser = argparse.ArgumentParser(description='Toxic bot runner')
	parser.add_argument('-k', '--key', help='API Key')
	args = parser.parse_args()
	bot.run(args.key)
