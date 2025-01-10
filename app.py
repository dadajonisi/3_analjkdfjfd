import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import pickle

# Bot token
BOT_TOKEN = "7765871087:AAGhd95YSRkNX4bZDGPljBOxxIfwF_h7Cg4"

# Ovozni xususiyatlarga ajratish funksiyasi
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Ovozni saqlash va solishtirish uchun ovozlar bazasi
VOICE_DATABASE = "voice_database.pkl"

def load_voice_database():
    if os.path.exists(VOICE_DATABASE):
        with open(VOICE_DATABASE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_voice_database(database):
    with open(VOICE_DATABASE, 'wb') as f:
        pickle.dump(database, f)

# Foydalanuvchi ovozini saqlash
voice_db = load_voice_database()

def identify_or_save_voice(user_id, features):
    if user_id in voice_db:
        # Foydalanuvchining ovozini solishtirish
        stored_features = voice_db[user_id]
        similarity = cosine_similarity([features], [stored_features])[0][0]
        if similarity > 0.8:
            return "Ovoz mos keldi!"
        else:
            return "Ovoz mos kelmadi."
    else:
        # Foydalanuvchining ovozini saqlash
        voice_db[user_id] = features
        save_voice_database(voice_db)
        return "Ovoz muvaffaqiyatli saqlandi!"

# Botga xush kelibsiz
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Assalomu alaykum! Ovoz orqali shaxsni identifikatsiya qiluvchi botga xush kelibsiz. Menga ovozli xabar yuboring.")

# Foydalanuvchi yuborgan ovozni qayta ishlash
def handle_audio(update: Update, context: CallbackContext) -> None:
    voice = update.message.voice.get_file()
    file_path = f"{voice.file_id}.ogg"
    voice.download(file_path)

    # Ovoz formatini o'zgartirish
    audio = AudioSegment.from_ogg(file_path)
    audio.export("audio.wav", format="wav")

    # Ovoz xususiyatlarini chiqarish
    features = extract_features("audio.wav")

    # Foydalanuvchi identifikatsiyasi yoki ovozni saqlash
    user_id = str(update.message.from_user.id)
    response = identify_or_save_voice(user_id, features)

    # Javob yuborish
    update.message.reply_text(response)

    # Foydalanuvchi ovozlarini saqlash
    os.remove(file_path)
    os.remove("audio.wav")

def main():
    updater = Updater(BOT_TOKEN)

    dispatcher = updater.dispatcher

    # Bot buyruqlari
    dispatcher.add_handler(CommandHandler("start", start))

    # Ovozni qabul qilish
    dispatcher.add_handler(MessageHandler(Filters.voice, handle_audio))

    # Botni ishga tushirish
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

# Qayerdan olish mumkin:
# Agar voice_database.pkl mavjud bo'lmasa, dastur uni ish vaqtida yaratadi.
# Ushbu fayl foydalanuvchilarning ovoz xususiyatlarini saqlash uchun ishlatiladi.
# Dastur birinchi marta ishlatilganda, ovozlar avtomatik ravishda ushbu faylga yoziladi.
