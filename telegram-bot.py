import logging
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, Filters, MessageHandler, InlineQueryHandler
from telegram.ext.dispatcher import run_async

import hashlib
import numpy as np
import copy, time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from read_csv import get_titles, get_most_poular, add_rating, get_md
from hybrid import final_res


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

telegram.constants.MAX_MESSAGE_LENGTH = 10000
telegram.constants.MAX_CAPTION_LENGTH = 1000

# dovro' fare un thread the ogni N secondi salva su file questo dizionario
# mi serve per i consigli della persona ideale
dict_name_id = {}

# text="*bold* _italic_ `fixed width font` [link](http://google.com)."
# bot.send_photo(chat_id=chat_id, photo='https://telegram.org/img/t_logo.png')

flag_input_start = False
titles_movies = get_titles()

# title deve essere in lowercase
# partial se vuole tutti i possibili risulati di quel tiolo, altrimenti resituisce un nome solo (se lo trova)
def search(title, partial=True):
    possible_original_title = []
    for t in titles_movies:
        t_copy = copy.deepcopy(t).lower()
        if title in t_copy:
            possible_original_title.append(t)

    if partial:
        return possible_original_title

    if possible_original_title:
        # prendo quello che ha la lunghezza piu' vicina a "title"
        original_title = possible_original_title[0]
        for t in possible_original_title:
            if abs(len(title)-len(t)) < abs(len(title)-len(original_title)):
                original_title = t
        return original_title
    return None


# creo l'interfaccia per tutti i possibili titoli
def input_movie_user(bot, update, possible_original_title, rate=0, info=False):
    keyboard = [[i] for i in range(len(possible_original_title))]
    count = 0
    if info:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(title, callback_data="5 " + title)
            count += 1
    elif rate == 0:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(title, callback_data="3 " + title)
            count += 1
    else:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(title, callback_data="4 " + title + " || " + str(rate))
            count += 1
    try:
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Which one?', reply_markup=reply_markup)
    except:
        chat_id = update.message.chat_id
        bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")
        bot.send_message(chat_id, 'Insert the name of the film', reply_markup=telegram.ForceReply(force_reply=True))
        flag_input_start = True

@run_async
def start(bot, update):
    movies = get_most_poular()
    keyboard = [[i] for i in range(len(movies)+1)]
    keyboard[0][0] = InlineKeyboardButton("Write the name manually", callback_data="2 ")
    
    count = 1
    for title, img in movies:
        keyboard[count][0] = InlineKeyboardButton(title, callback_data="1 " + title)
        count += 1

    reply_markup = InlineKeyboardMarkup(keyboard)
    name = str(update['message']['chat']['username'])
    id_name = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)
    dict_name_id[name] = id_name

    # update.message.reply_text('*Hi* @' + name + "\nYour id is: `" + str(id_name) + "`", parse_mode=telegram.ParseMode.MARKDOWN)
    update.message.reply_text('*Hi* @' + name, parse_mode=telegram.ParseMode.MARKDOWN)

    update.message.reply_text('Click your favorite movie or search the name', reply_markup=reply_markup)


# def contact(bot, update):
#     contact_keyboard = telegram.KeyboardButton(text="send_contact", request_contact=True)
#     custom_keyboard = [[contact_keyboard]]
#     reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
#     chat_id = update.message.chat_id
#     bot.send_message(chat_id=chat_id, 
#                      text="Would you mind sharing your contact with me?", 
#                      reply_markup=reply_markup)

# quando clicco un qualunque bottone
@run_async
def button(bot, update):
    query = update.callback_query
    chat_id = query.message.chat_id
    option = str(query.data.split(' ')[0])

    # START
    if option == "1":
        title = ' '.join(query.data.split(' ')[1:]).lower()
        user_name = str(update['callback_query']['from_user']['username'])
        if len(user_name) < 5:
            bot.send_message(chat_id=chat_id, text="\t-- !ERROR! --\nYour data won't be save\nYou don't have a username set on Telegram"\
                             "[How to do it](https://telegram.org/blog/usernames-and-secret-chats-v2)")

        original_title = search(title, partial=False)

        if original_title:
            bot.edit_message_text(text=f"Selected option: {original_title}",
                                  chat_id=chat_id, message_id=query.message.message_id)

            add_rating(user_name, original_title, 5)

            bot.send_message(chat_id=chat_id, text="Wait for the recommandation")

            # RECOMMANDATION
            recommanded_movies = final_res(str(user_name))

            keyboard = [[i] for i in range(len(recommanded_movies))]
            count = 0
            for key, value in recommanded_movies:
                # key = ''.join([i if ord(i) < 128 else '~' for i in key])
                keyboard[count][0] = InlineKeyboardButton(str(key) + " -> " + str(value)[:5], callback_data="3 " + key)
                count += 1
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            bot.send_message(chat_id=chat_id, text='Click your favorite movie', reply_markup=reply_markup)

        else:
            bot.edit_message_text(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]",
                                  message_id=query.message.message_id)

    # INPUT USER
    elif option == "2":
        global flag_input_start
        bot.send_message(chat_id, 'Insert the name of the film', reply_markup=telegram.ForceReply(force_reply=True))
        flag_input_start = True

    # SHOW MOVIE AND RATING
    elif option == "3":
        title = ' '.join(query.data.split(' ')[1:])
        user_name = str(update['callback_query']['from_user']['username'])

        bot.edit_message_text(text=f"Selected option: {title}",
                              chat_id=chat_id, message_id=query.message.message_id)

        keyboard = [
                     [
                        InlineKeyboardButton("1", callback_data="4 " + title + " || 1"),
                        InlineKeyboardButton("2", callback_data="4 " + title + " || 2"), 
                        InlineKeyboardButton("3", callback_data="4 " + title + " || 3")
                     ],
                     [
                        InlineKeyboardButton("4", callback_data="4 " + title + " || 4"), 
                        InlineKeyboardButton("5", callback_data="4 " + title + " || 5")
                     ]
                   ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text='Insert your personal rating for the film', reply_markup=reply_markup)
    
    # SAVE AND ASK AGAIN
    elif option == "4":
        option2 = ' '.join(query.data.split(' ')[1:])
        user_name = str(update['callback_query']['from_user']['username'])
        title, rating = option2.split(' || ')

        bot.edit_message_text(text=f"{int(rating)}/5 for {title}",
                              chat_id=chat_id, message_id=query.message.message_id)

        add_rating(user_name, title, int(rating))

        bot.send_message(chat_id=chat_id, text="Wait for the recommandation")

        recommanded_movies = final_res(str(user_name))

        keyboard = [[i] for i in range(len(recommanded_movies))]
        count = 0
        for key, value in recommanded_movies:
            # key = ''.join([i if ord(i) < 128 else '~' for i in key])
            keyboard[count][0] = InlineKeyboardButton(str(key) + " -> " + str(value)[:5], callback_data="3 " + key)
            count += 1

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text='Click your favorite movie', reply_markup=reply_markup)

    # INFO MOVIE
    elif option == "5":
        user_name = str(update['callback_query']['from_user']['username'])
        title = ' '.join(query.data.split(' ')[1:])

        md = get_md()
        row_title = md.loc[md['title'] == title]

        # adult,belongs_to_collection,budget,genres,homepage,id,
        # imdb_id,original_language,original_title,overview,popularity,
        # poster_path,production_companies,production_countries,release_date,
        # revenue,runtime,spoken_languages,status,tagline,title,video,vote_average,vote_count

        message = "*" + str(row_title['original_title'].values[0]).upper() + "* \n" + \
                  "*Release Date*: " + str(row_title['release_date'].values[0]) + "\n" #+ "[How to do it](https://telegram.org/blog/usernames-and-secret-chats-v2)"
                #   "*Genres*: " + ','.join([row_title['genres'].values[i] for i in range(len(row_title['genres']))]) + "\n" 
                #   "*Runtime*: " + str(row_title['runtime'].values[0]) + " minuts\n"
                #   "*Overview*:\n" + str(row_title['overview'].values[0])
                

        bot.edit_message_text(chat_id=chat_id, text=message, message_id=query.message.message_id, parse_mode=telegram.ParseMode.MARKDOWN)
        bot.send_photo(chat_id=chat_id, photo="https://image.tmdb.org/t/p/original/" + str(row_title['poster_path'].values[0]))

@run_async        
def input_user(bot, update):
    global flag_input_start
    chat_id = update.message.chat_id

    if flag_input_start == True:
        flag_input_start = False
        title = str(update.message.text).lower()
        user_name = str(update['message']['chat']['username'])

        possible_original_title = search(title)

        if possible_original_title:
            input_movie_user(bot, update, possible_original_title)
        else:
            bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def rate_movie(bot, update, args):
    chat_id = update.message.chat_id
    if len(args) < 1:
        bot.send_message(chat_id=chat_id, text="Invalid argument: /rate (name) [rate]\nExample:\n\t/rate Interstellar")
    else:
        title, rate = '', ''
        if args[-1].isdigit() and int(args[-1]) in [1, 2, 3, 4, 5]:
            title, rate = ' '.join(args[:-1]), args[-1]
        else:
            title = ' '.join(args)

        possible_original_title = search(title.lower())
        if len(possible_original_title):
            if rate:
                input_movie_user(bot, update, possible_original_title, rate)
            else:
                input_movie_user(bot, update, possible_original_title)
        else:
            bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def search_movie(bot, update, args):
    chat_id = update.message.chat_id
    if len(args) < 1:
        bot.send_message(chat_id=chat_id, text="Invalid argument: /search name\nExample:\n\t/search Interstellar")
    else:
        title = ' '.join(args)
        possible_original_title = search(title.lower())
        if len(possible_original_title):
            input_movie_user(bot, update, possible_original_title, info=True)
        else:
            bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def list_movies(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    recommanded_movies = final_res(str(user_name))

    if len(recommanded_movies) < 1:
        bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
        return 

    keyboard = [[i] for i in range(len(recommanded_movies))]
    count = 0
    for key, value in recommanded_movies:
        keyboard[count][0] = InlineKeyboardButton(str(key) + " -> " + str(value)[:5], callback_data="3 " + key)
        count += 1

    reply_markup = InlineKeyboardMarkup(keyboard)
    bot.send_message(chat_id=chat_id, text='Here it is your recommanded movies :)\nClick your favorite movie', reply_markup=reply_markup)

@run_async
def inline_movies(bot, update):
    user_name = str(update['inline_query']['from_user']['username'])
    chat_id = update.inline_query.from_user.id

    query = update.inline_query.query
    results = list()

    if query == 'movies' or query == "Movies":
        recommanded_movies = final_res(str(user_name))
        # return_msg = ""
        # for key, value in recommanded_movies:
        #     return_msg += str(key) + " -> " + str(value)[5:] + "\n" 
        keyboard = [[i] for i in range(len(recommanded_movies))]
        count = 0
        for key, value in recommanded_movies:
            keyboard[count][0] = InlineKeyboardButton(str(key) + " -> " + str(value)[:5], callback_data="3 " + key)
            count += 1

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text='Here it is your recommanded movies :)', reply_markup=reply_markup)

        results.append(
            InlineQueryResultArticle(
                id=chat_id,
                title='Movies',
                input_message_content=InputTextMessageContent('Click your favorite movie'),
                reply_markup=reply_markup
            )
        )
    elif query == 'search':
        return
    elif query == 'rate':
        return
    else:
        return
    
    bot.answer_inline_query(update.inline_query.id, results)



def help(bot, update):
    update.message.reply_text("Use /start to test this bot.")

from telegram.error import (TelegramError, Unauthorized, BadRequest, 
                            TimedOut, ChatMigrated, NetworkError)
def error_callback(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)
    try:
        raise error
    except Unauthorized:
        print("Unauthorized")
        # remove update.message.chat_id from conversation list
    except BadRequest as e:
        print("BadRequest")
        print(e)
        # handle malformed requests - read more below!
    except TimedOut:
        print("TimedOut")
        # handle slow connection problems
    except NetworkError:
        print("NetworkError")
        # handle other connection problems
    except ChatMigrated as e:
        print("ChatMigrated")
        print(e)
        # the chat_id of a group has changed, use e.new_chat_id instead
    except TelegramError:
        print("TelegramError")
        # handle all other telegram related errors


# bot = Bot(TOKEN)
# update_queue = Queue()
# dp = Dispatcher(bot, update_queue)

updater = Updater("680393052:AAHmCBJlFi3u326GMIKqEDuazKJvH9MpFm4", workers=32)

ud = updater.dispatcher
ud.add_handler(CommandHandler('start', start))
ud.add_handler(CommandHandler('movies', list_movies))
ud.add_handler(CommandHandler('rate', rate_movie, pass_args=True))
ud.add_handler(CommandHandler('search', search_movie, pass_args=True))
ud.add_handler(CallbackQueryHandler(button)) # , pattern='main'
# ud.add_handler(CallbackQueryHandler(info_film))
ud.add_handler(InlineQueryHandler(inline_movies))
ud.add_handler(MessageHandler(Filters.text, input_user))
ud.add_handler(CommandHandler('help', help))
ud.add_error_handler(error_callback)

updater.start_polling(timeout=100.0)
updater.idle()