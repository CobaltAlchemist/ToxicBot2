import pandas as pd
import numpy as np
import os
from discord import Guild, User, TextChannel, Message
from .constants import *
from asyncio import Lock
import sqlite3



class DiscordDatabase:
	def __init__(self, directory : str):
		self.directory = directory
		runscript = True
		self.dbcon = sqlite3.connect('toxic.db')
		self.dbcursor = self.dbcon.cursor()
		with open('schema_script.sql') as script:
			self.dbcursor.executescript(script.read())
		self.writelock = Lock()
		self.usercols = ['incidents', 'total_seen'] + TOXCLASSES_ORIG

	def user_data(self, user : User, guild : Guild = None):
		s = "SELECT " + self._all_usercols(guild is None)
		s += f" FROM users WHERE author={user.id} "
		if guild is None:
			s += " GROUP BY guild "
		else:
			s += f" AND guild={guild.id} "
		data = self.dbcursor.execute(s).fetchone()
		return {k: v for k, v in zip(self.usercols, data)}

	def guild_data(self, guild : Guild):
		s = "SELECT " + self._all_usercols(True) + \
			f"FROM users WHERE guild={guild.id} GROUP BY guild"
		data = self.dbcursor.execute(s).fetchone()
		return {k: v for k, v in zip(self.usercols, data)}

	def last_message(self, guild : Guild, channel : TextChannel, user : User):
		s = f"SELECT text FROM last_messages WHERE guild={guild.id} AND channel={channel.id} AND author={user.id}"
		data = self.dbcursor.execute(s).fetchone()
		if data is None:
			return None
		return data[0]

	async def add_message(self, message : Message, toxpreds):
		async with self.writelock:
			fulluser = DiscordDatabase._full_user(message.author)
			self.dbcon.execute("INSERT OR IGNORE INTO users (guild, author) VALUES (?, ?)", (message.guild.id, message.author.id))
			self.dbcon.execute(f"UPDATE users SET total_seen = total_seen + 1 WHERE guild={message.guild.id} AND author={message.author.id}")
			if max(toxpreds) > 0.5:
				execstr = f"UPDATE users SET incidents = incidents + 1, " + \
					", ".join([f"{c} = {c} + 1" for _, c in filter(lambda x: x[0] > 0.5, zip(toxpreds, TOXCLASSES_ORIG))]) + \
					f" WHERE guild={message.guild.id} AND author={message.author.id}"
				self.dbcon.execute(execstr)
				self.dbcon.execute("INSERT INTO history (guild, author, " + ', '.join(TOXCLASSES_ORIG) +\
				   ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (message.guild.id, message.author.id, *[str(x) for x in toxpreds]))
			self.dbcon.execute("INSERT OR IGNORE INTO last_messages (guild, author, channel) VALUES (?, ?, ?)",
					  (message.guild.id, message.author.id, message.channel.id))
			execstr = f'UPDATE last_messages SET text="{message.content}" ' + \
				f"WHERE guild={message.guild.id} AND author={message.author.id} AND channel={message.channel.id}"
			self.dbcon.execute(execstr)
			self.dbcon.commit()

	def _all_usercols(self, agg = False):
		cols = self.usercols
		if agg:
			cols = [f"SUM({c})" for c in cols]
		return ', '.join(cols)

	@staticmethod
	def _full_user(user : User):
		return f"{user.name}#{user.discriminator}"
