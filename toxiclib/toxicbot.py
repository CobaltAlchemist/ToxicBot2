import discord
import numpy as np
import json
from random import choice, uniform
from discord.ext import commands
from .predictor import Predictor
from .ddb import DiscordDatabase
from .constants import *
	
# For main autonomous behavior
class ToxicCog(commands.Cog, name="Main Commands"):
	def __init__(self, bot: commands.Bot, database : DiscordDatabase, config, model_dir='.', device='cpu', **kwargs):
		self.bot = bot
		self.predictor = Predictor(model_dir, device)
		self.db = database
		self.config = config
		
	@commands.Cog.listener()
	async def on_ready(self):
		print("Logging in as", self.bot.user.name, self.bot.user.id)

	@commands.Cog.listener()
	async def on_message(self, message: discord.Message):
		if message.author.id == self.bot.user.id:
			return
		if message.guild is None:
			return
		if len(message.content) == 0:
			return
		if message.content[0] == self.bot.command_prefix:
			return message.delete()
		await self.user_messages(message)
		preds = self.predictor(message.content)
		await self.db.add_message(message, preds)
		if np.sum(preds > 0.5) >= 4:
			await message.channel.send(choice(self.config['toxic']))
		elif np.sum(preds > 0.5) >= 2 and preds[3] > 0.5:
			await message.add_reaction('<pepocringe:746992858535297044>')
		elif np.sum(preds > 0.5) >= 2:
			await message.add_reaction('<cathands:755262972204679279>')
		if 'sorry' in message.content:
			await message.channel.send(choice(self.config['apology']))

	@commands.Cog.listener()
	async def on_reaction_add(self, reaction, user):
		message = reaction.message
		if message.author.id != self.bot.user.id:
			return
	
		if reaction.emoji in ['🎖', '🏅', '🥇'] and uniform(0, 1) < 0.1:
			await message.channel.send('Thanks for the gold, kind stranger!')

	@commands.command()
	async def rate(self, ctx: commands.Context, *s : str):
		ratings = self.predictor(' '.join(s))
		print(' '.join(s))
		fmt = '\n'.join([f"\t{c}: {p*100:.2f}%" for c, p in zip(TOXCLASSES_HUMAN, ratings)])
		await ctx.send(f"I rate the message:\n{fmt}")

	@commands.command()
	async def why(self, ctx: commands.Context):
		msg = ctx.message
		user = msg.author
		channel = msg.channel
		guild = msg.guild
		last_message = self.db.last_message(guild, channel, user)
		if last_message is None:
			await ctx.send(f"Sorry {user.name}, I didn't find your last message here")
			return
		preds = self.predictor(last_message)
		if len(last_message) > 100:
			last_message = last_message[:97] + "..."
		s = f"Your last message, {user.name} '{last_message}' was rated:\n\t"
		s += '\n\t'.join([f"{c}: {p*100:.2f}%" for c, p in zip(TOXCLASSES_HUMAN, preds)])
		await ctx.send(s)

	@commands.command()
	async def stats(self, ctx: commands.Context):
		s = ""
		for user in ctx.message.mentions:
			stats = self.db.user_data(user, ctx.message.guild)
			s += self._format_stats(user.name, stats)
		if not ctx.message.mentions:
			stats = self.db.guild_data(ctx.message.guild)
			s += self._format_stats(ctx.message.guild.name, stats)
		await ctx.send(s.rstrip())

	@commands.command()
	async def tokens(self, ctx: commands.Context, *s):
		s = ' '.join(s)
		toks = self.predictor.tokenize_text(s)
		await ctx.send(f"Got the following tokens for that: {', '.join(toks)}")

	@commands.command()
	async def ping(self, ctx: commands.Context):
		await ctx.send(f"Pong! {round(self.bot.latency * 1000)}ms")

	async def user_messages(self, message : discord.Message):
		user_config = self.config['user']
		resps = []
		resps.extend(user_config['all'])
		p = user_config['p_default']
		id_str = str(message.author.id)
		if id_str in user_config:
			user_config = user_config[id_str]
			if 'p' in user_config:
				p = user_config['p']
			resps.extend(user_config['resps'])
		if uniform(0, 1) < p:
			await message.channel.send(choice(resps))
			return True
		return False

	def _format_stats(self, name, stats):
		s = ""
		totals = np.array([stats[c] for c in TOXCLASSES_ORIG])
		s += f"For {name} I found the following:\n" \
			f"\t Toxic incidents: {stats['incidents']} / {stats['total_seen']} aka {100 * stats['incidents'] / stats['total_seen']:.2f}% rate\n"\
			f"\tFavorite form of toxicity (unweighted... for now): {TOXCLASSES_HUMAN[np.argmax(totals)]}\n"\
			f"\tIncidents by category:\n\t\t"
		s += '\n\t\t'.join([f"{c}: {v}" for c, v in zip(TOXCLASSES_HUMAN, totals)]) + "\n"
		return s

def ToxicBot(prefix='%', **kwargs):
	bot = commands.Bot(
			prefix,
			case_insensitive=True,
			description="Bot for identifying toxic comments",
			activity=discord.Game("TB 2.0! | %help")
		)
	db = DiscordDatabase('db')
	with open('resp_config.json') as f:
		config = json.load(f)
	bot.add_cog((ToxicCog(bot, db, config, **kwargs)))
	return bot
