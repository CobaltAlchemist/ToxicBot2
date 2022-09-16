import discord
import numpy as np
import json
from random import choice, uniform
from discord.ext import commands
from .ddb import DiscordDatabase
from .constants import *
from asyncio import Lock
import base64
from io import BytesIO
from PIL import Image
import requests
import argparse
import os

# For main autonomous behavior


class ToxicCog(commands.Cog, name="Main Commands"):
    def __init__(self, bot: commands.Bot, database: DiscordDatabase, config, text_model_url, **kwargs):
        self.bot = bot
        self.db = database
        self.config = config
        self.text_model_url = 'http://' + text_model_url + '/'

    @commands.Cog.listener()
    async def on_ready(self):
        print("Logging in as", self.bot.user.name, self.bot.user.id)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.id == self.bot.user.id:
            return
        if message.author.bot:
            return
        if message.guild is None:
            return
        if len(message.content) == 0:
            return
        if message.content[0] == self.bot.command_prefix:
            return

        if message.content.lower().startswith("toxic show me ") or message.content.lower().startswith("toxicbot show me "):
            idx = message.content.index("show me ")
            s = message.content[idx+len("show me "):]
            await self.send_stable_diffusion({'prompt': s}, message.channel)
            return

        await self.user_messages(message)
        preds = await self.toxic_predict(message.content)
        if preds is None:
            return
        identity_hate = preds['Identity Attack'] > 0.5
        preds = np.array([v for _, v in preds.items()])
        print(preds)

        await self.db.add_message(message, preds)
        if np.sum(preds > 0.5) >= 4:
            await message.channel.send(choice(self.config['toxic']))
        elif np.sum(preds > 0.5) >= 2 and identity_hate:
            await message.add_reaction('<pepocringe:746992858535297044>')
        elif np.sum(preds > 0.5) >= 2:
            await message.add_reaction('<cathands:755262972204679279>')
        if 'sorry' in message.content.lower().split() and 'toxic' in message.content.lower():
            await message.channel.send(choice(self.config['apology']))

    async def send_dalle(self, prompt, channel):
        print(f"Got dalle request '{prompt}'")
        async with self.model_lock:
            async with channel.typing():
                print("Sending...'")
                try:
                    r = requests.post(
                        'http://192.168.0.15:10001/plugin/dalle_plugin', json={"text": prompt})
                except requests.exceptions.ConnectionError:
                    print("Can't reach dalle")
                    await channel.send("I can't reach that right now")
                    return
                if not r.ok:
                    print("Got bad response from dalle")
                    await channel.send("Dalle's brain isn't working so good right now")
                    return
                image = Image.open(BytesIO(base64.b64decode(
                    r.json()['response']['image'].encode('utf-8'))))
                image.save('temp.png')
                await channel.send(file=discord.File('temp.png'))

    async def send_ldm(self, req, channel):
        print(f"Got ldm request '{req}'")
        async with self.model_lock, channel.typing():
            print("Sending...'")
            try:
                r = requests.post(
                    'http://192.168.0.15:10001/plugin/diffusion_plugin', json=req)
            except requests.exceptions.ConnectionError:
                print("Can't reach ldm")
                await channel.send("I can't reach that right now")
                return
            if not r.ok:
                print("Got bad response from dalle")
                await channel.send("LDM's brain isn't working so good right now")
                return
            response = r.json()['response']
            if response['nsfw']:
                await channel.send("😳 Sorry but I can't show you this!")
                return
            image = Image.open(BytesIO(base64.b64decode(
                response['image'].encode('utf-8'))))
            image.save('temp.png')
            await channel.send(file=discord.File('temp.png'))

    async def send_stable_diffusion(self, req, channel):
        print(f"Got sd request '{req}'")
        server = os.environ["STABLE_DIFFUSION_SERVER"]
        async with self.model_lock, channel.typing():
            print("Sending...'")
            try:
                r = requests.post(
                    f'http://{server}/predict', json=req)
            except requests.exceptions.ConnectionError:
                print("Can't reach sd")
                await channel.send("I can't reach that right now")
                return
            if not r.ok:
                print("Got bad response from sd")
                await channel.send("SD's brain isn't working so good right now")
                return
            response = r.json()['response']
            if response['nsfw']:
                await channel.send("😳 Sorry but I can't show you this!")
                return
            image = Image.open(
                BytesIO(base64.b64decode(response['image'].encode('utf-8'))))
            image.save('temp.png')
            await channel.send(file=discord.File('temp.png'))

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        message = reaction.message
        if message.author.id != self.bot.user.id:
            return

        if reaction.emoji in ['🎖', '🏅', '🥇'] and uniform(0, 1) < 0.1:
            await message.channel.send('Thanks for the gold, kind stranger!')

    @commands.command()
    async def rate(self, ctx: commands.Context, *s: str):
        preds = await self.toxic_predict(' '.join(s))
        if preds is None:
            await ctx.send("My brain hurts")
            return
        fmt = '\n'.join([f"\t{c}: {p*100:.2f}%" for c, p in preds.items()])
        await ctx.send(f"I rate the message:\n{fmt}")

    async def request(self, url, j):
        try:
            r = requests.post(url, json=j)
        except requests.exceptions.ConnectionError:
            return None
        if not r.ok:
            return None
        return r.json()

    async def toxic_predict(self, s: str):
        return await self.request(self.text_model_url + 'predict', {'text': s})

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
        preds = await self.toxic_predict(last_message)
        if len(last_message) > 100:
            last_message = last_message[:97] + "..."
        s = f"Your last message, {user.name} '{last_message}' was rated:\n\t"
        s += '\n\t'.join([f"{c}: {p*100:.2f}%" for c, p in preds.items()])
        await ctx.send(s)

    @commands.command()
    async def echo(self, ctx: commands.Context, *s: str):
        await ctx.send(' '.join(s))

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
        toks = await self.request(self.text_model_url + 'tokens', {'text': s})
        if toks is None:
            await ctx.send("Oof owie my brain")
            return
        toks = toks['tokens']
        await ctx.send(f"Got the following tokens for that: {toks}")

    @commands.command()
    async def dalle(self, ctx: commands.Context, *s):
        s = ' '.join(s)
        await self.send_dalle(s, ctx.channel)

    @commands.command()
    async def ldm(self, ctx: commands.Context, *s):
        s = ' '.join(s)
        await self.send_ldm({'prompt': s}, ctx.channel)

    @commands.command()
    async def sd(self, ctx: commands.Context, *s):
        s = ' '.join(s)
        await self.send_stable_diffusion({'prompt': s}, ctx.channel)

    @commands.command()
    async def ldm_e(self, ctx: commands.Context, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-H', type=int, default=256)
        parser.add_argument('-W', type=int, default=256)
        parser.add_argument('-s', '--scale', type=float, default=5.0)
        parser.add_argument('-n', '--n_steps', type=int, default=30)
        parser.add_argument('prompt', type=str, nargs='+')
        try:
            opt = parser.parse_args(args)
        except:
            await ctx.send(parser.format_help())
            return
        if opt.H * opt.W > 524288 or opt.n_steps > 250:
            await ctx.send("Hey, chill. Keep the pixels under 500k and steps under 100")
            return
        r = {'prompt': ' '.join(opt.prompt),
             'h': opt.H,
             'w': opt.W,
             'scale': opt.scale,
             'ddim_steps': opt.n_steps,
             }
        await self.send_ldm(r, ctx.channel)

    @commands.command()
    async def ping(self, ctx: commands.Context):
        await ctx.send(f"Pong! {round(self.bot.latency * 1000)}ms")

    async def user_messages(self, message: discord.Message):
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
            if resps and uniform(0, 1) < p:
                await message.channel.send(choice(resps))
                return True
        return False

    def _format_stats(self, name, stats):
        s = ""
        totals = np.array([stats[c] for c in TOXCLASSES_ORIG])
        s += f"For {name} I found the following:\n" \
            f"\t Toxic incidents: {stats['incidents']}/{stats['total_seen']}, {100 * stats['incidents'] / stats['total_seen']:.1f}% toxic\n"\
            f"\tFavorite form of toxicity (unweighted... for now): {TOXCLASSES_HUMAN[np.argmax(totals)]}\n"\
            f"\tIncidents by category:\n\t\t"
        s += '\n\t\t'.join([f"{c}: {v}" for c,
                           v in zip(TOXCLASSES_HUMAN, totals)]) + "\n"
        return s


def ToxicBot(data_dir, tm_url, prefix='%', **kwargs):
    bot = commands.Bot(
        prefix,
        case_insensitive=True,
        description="Bot for identifying toxic comments",
        activity=discord.Game("TB 2.0! | %help")
    )
    db = DiscordDatabase(data_dir)
    with open(os.path.join(data_dir, 'resp_config.json')) as f:
        config = json.load(f)
    bot.add_cog((ToxicCog(bot, db, config, tm_url, **kwargs)))
    return bot
