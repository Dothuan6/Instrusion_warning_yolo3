import telegram

async def send_telegram(photo_path='alert.png'):
    try:
        my_token = "6731624798:AAG0CR_mCMgpVt62ZNUfF9360N0k02Z1cok"
        bot = telegram.Bot(token=my_token)
        with open(photo_path, 'rb') as photo:
            await bot.send_photo(chat_id=5614539775, photo=photo, caption='Nguy hiểm có xâm nhập!')
        print('Send to telegram successfully!')
    except Exception as e:
        print(e)
