from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import json
from PIL import Image
from constant import *
import numpy as np
import io
from utils import *
import faiss
import numpy as np
import torch
import torchvision.models as models
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(YSY_COMIC_TEXT_PATH, 'r', encoding="utf-8") as f:
    chapter_text = f.read()

# ========== Patch 800 ==========
with open(YSY_COMIC_800_EMBED_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_800_embed_resnext = json.load(f)

resnext_800_embed_indices_mapping = [chapter_800_embed_resnext[i]['file'][13:17] for i in range(len(chapter_800_embed_resnext))]
embeddings = np.array([chapter_800_embed_resnext[i]['embedding'] for i in range(len(chapter_800_embed_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_800_embed_database = faiss.IndexFlatIP(2048)
resnext_800_embed_database.add(embeddings)
print('Created database ResNeXt Embed with patch size 800')
del chapter_800_embed_resnext, embeddings
# ===============================
with open(YSY_COMIC_800_SOBEL_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_800_sobel_resnext = json.load(f)

resnext_800_sobel_indices_mapping = [chapter_800_sobel_resnext[i]['file'][13:17] for i in range(len(chapter_800_sobel_resnext))]
embeddings = np.array([chapter_800_sobel_resnext[i]['embedding'] for i in range(len(chapter_800_sobel_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_800_sobel_database = faiss.IndexFlatIP(2048)
resnext_800_sobel_database.add(embeddings)
print('Created database ResNeXt Sobel with patch size 800')
del chapter_800_sobel_resnext, embeddings
# ===============================
with open(YSY_COMIC_800_LAPLACE_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_800_laplace_resnext = json.load(f)

resnext_800_laplace_indices_mapping = [chapter_800_laplace_resnext[i]['file'][13:17] for i in range(len(chapter_800_laplace_resnext))]
embeddings = np.array([chapter_800_laplace_resnext[i]['embedding'] for i in range(len(chapter_800_laplace_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_800_laplace_database = faiss.IndexFlatIP(2048)
resnext_800_laplace_database.add(embeddings)
print('Created database ResNeXt Laplace with patch size 800')
del chapter_800_laplace_resnext, embeddings

# ========== Patch 266 ==========
with open(YSY_COMIC_266_EMBED_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_266_embed_resnext = json.load(f)

resnext_266_embed_indices_mapping = [chapter_266_embed_resnext[i]['file'][13:17] for i in range(len(chapter_266_embed_resnext))]
embeddings = np.array([chapter_266_embed_resnext[i]['embedding'] for i in range(len(chapter_266_embed_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_266_embed_database = faiss.IndexFlatIP(2048)
resnext_266_embed_database.add(embeddings)
print('Created database ResNeXt Embed with patch size 266')
del chapter_266_embed_resnext, embeddings
# ===============================
with open(YSY_COMIC_266_SOBEL_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_266_sobel_resnext = json.load(f)

resnext_266_sobel_indices_mapping = [chapter_266_sobel_resnext[i]['file'][13:17] for i in range(len(chapter_266_sobel_resnext))]
embeddings = np.array([chapter_266_sobel_resnext[i]['embedding'] for i in range(len(chapter_266_sobel_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_266_sobel_database = faiss.IndexFlatIP(2048)
resnext_266_sobel_database.add(embeddings)
print('Created database ResNeXt Sobel with patch size 266')
del chapter_266_sobel_resnext, embeddings
# ===============================
with open(YSY_COMIC_266_LAPLACE_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_266_laplace_resnext = json.load(f)

resnext_266_laplace_indices_mapping = [chapter_266_laplace_resnext[i]['file'][13:17] for i in range(len(chapter_266_laplace_resnext))]
embeddings = np.array([chapter_266_laplace_resnext[i]['embedding'] for i in range(len(chapter_266_laplace_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_266_laplace_database = faiss.IndexFlatIP(2048)
resnext_266_laplace_database.add(embeddings)
print('Created database ResNeXt Laplace with patch size 266')
del chapter_266_laplace_resnext, embeddings

# ========== ResNeXt model ==========
resnext_model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
resnext_model = torch.nn.Sequential(*list(resnext_model.children())[:-1])  # Remove final layer
resnext_model = resnext_model.to(device)
resnext_model.eval()
resnext_preprocess = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2.transforms()
print('Loaded ResNeXt model')
    
# Vision Transformer
with open(YSY_COMIC_800_EMBED_VIT, 'r', encoding="utf-8") as f:
    chapter_800_embed_vit = json.load(f)

vit_800_embed_indices_mapping = [chapter_800_embed_vit[i]['file'][13:17] for i in range(len(chapter_800_embed_vit))]
embeddings = np.array([chapter_800_embed_vit[i]['embedding'] for i in range(len(chapter_800_embed_vit))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
vit_800_embed_database = faiss.IndexFlatL2(1280)
vit_800_embed_database.add(embeddings)
print('Created database ViT with patch size 800')
del chapter_800_embed_vit, embeddings

# with open(YSY_COMIC_266_EMBED_VIT, 'r', encoding="utf-8") as f:
#     chapter_266_embed_vit = json.load(f)

# vit_266_indices_mapping = [chapter_266_embed_vit[i]['file'][13:17] for i in range(len(chapter_266_embed_vit))]
# embeddings = np.array([chapter_266_embed_vit[i]['embedding'] for i in range(len(chapter_266_embed_vit))])
# embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# vit_266_database = faiss.IndexFlatL2(1280)
# vit_266_database.add(embeddings)
# print('Created database ViT with patch size 266')
# del chapter_266_embed_vit, embeddings

vit_model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
vit_model.heads = torch.nn.Identity()
vit_model = vit_model.to(device)
vit_model.eval()
vit_preprocess = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
print('Loaded ViT model')
    
with open(CHANNEL_MESSAGE_PATH, 'r') as f:
    message_data = json.load(f)
    
# Suppor functions
def get_message(chapter_id):
    for message in message_data:
        try:
            for attribute in message['media']['document']['attributes']:
                try:
                    if attribute['_'] == 'DocumentAttributeFilename' and chapter_id in attribute['file_name']:
                        return message['id']
                except:
                    continue
        except:
            continue
    return None

def search_chapter_by_text(text, top_n=5):
    print(text)
    # Matching score with OCR
    results = [(chapter[:4], get_text_matching_score(text, chapter)) for chapter in chapter_text.split('='*40+'\n') if chapter[:4].isnumeric()]
    
    results.sort(key=lambda x: (x[1], x[0]), reverse=True)
    scores = [score for _, score in results]
    chapter_ids = [id for id, _ in results]

    scores = scores[:top_n]
    scores = scores / np.sum(scores)
    chapter_ids = chapter_ids[:top_n]
    
    response = []
    for i in range(top_n):
        response.append((chapter_ids[i], scores[i], get_message(chapter_ids[i])))

    return response

def process_search_results(embedding, database, indices_mapping, sort_reverse=False):
    k = 30
    distances, indices = database.search(embedding, k)
    distances = distances.flatten()
    indices = indices.flatten()
    results = [(indices_mapping[id], d) for d, id in zip(distances, indices)]

    refined_results = dict()
    for chapter_id, distance in results:
        if chapter_id not in refined_results.keys():
            refined_results[chapter_id] = distance
        else:
            refined_results[chapter_id] = min(refined_results[chapter_id], distance) \
                                          if sort_reverse else max(refined_results[chapter_id], distance)
    refined_results = [(chapter_id, refined_results[chapter_id]) for chapter_id in refined_results.keys()] 
    refined_results.sort(key=lambda x: x[1], reverse=sort_reverse)
    return refined_results

def search_chapter_by_image(template, top_n=5):
    w, h = template.size

    img = resnext_preprocess(template).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnext_model(img).squeeze().unsqueeze(0).cpu().numpy()
    resnext_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    resnext_embed_results = None
    resnext_sobel_results = None
    resnext_laplace_results = None
    vit_embed_results = None

    if w * h < 400*400:
        resnext_embed_results = process_search_results(resnext_embedding, resnext_266_embed_database, 
                                                       resnext_266_embed_indices_mapping, 
                                                       sort_reverse=True)
        resnext_sum = sum([d**2 for _, d in resnext_embed_results])
        resnext_embed_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_embed_results]
        tmp = resnext_embed_results[:top_n//5]
        s = sum([d**2 for _, d in tmp])
        tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        print('ResNeXt 266 Embed results:', tmp)

        resnext_sobel_results = process_search_results(resnext_embedding, resnext_266_sobel_database, 
                                                       resnext_266_sobel_indices_mapping, 
                                                       sort_reverse=True)
        resnext_sum = sum([d**2 for _, d in resnext_sobel_results])
        resnext_sobel_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_sobel_results]
        tmp = resnext_sobel_results[:top_n//5]
        s = sum([d**2 for _, d in tmp])
        tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        print('ResNeXt 266 Sobel results:', tmp)

        resnext_laplace_results = process_search_results(resnext_embedding, resnext_266_laplace_database, 
                                                         resnext_266_laplace_indices_mapping, 
                                                         sort_reverse=True)
        resnext_sum = sum([d**2 for _, d in resnext_laplace_results])
        resnext_laplace_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_laplace_results]
        tmp = resnext_laplace_results[:top_n//5]
        s = sum([d**2 for _, d in tmp])
        tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        print('ResNeXt 266 Laplace results:', tmp)
    else:
        # resnext_embed_results = process_search_results(resnext_embedding, resnext_800_embed_database,
        #                                                resnext_800_embed_indices_mapping, 
        #                                                sort_reverse=True)
        # resnext_sum = sum([d**2 for _, d in resnext_embed_results])
        # resnext_embed_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_embed_results]
        # tmp = resnext_embed_results[:top_n//5]
        # s = sum([d**2 for _, d in tmp])
        # tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        # print('ResNeXt 800 Embed results:', tmp)

        # resnext_sobel_results = process_search_results(resnext_embedding, resnext_800_sobel_database,
        #                                                resnext_800_sobel_indices_mapping, 
        #                                                sort_reverse=True)
        # resnext_sum = sum([d**2 for _, d in resnext_sobel_results])
        # resnext_sobel_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_sobel_results]
        # tmp = resnext_sobel_results[:top_n//5]
        # s = sum([d**2 for _, d in tmp])
        # tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        # print('ResNeXt 800 Sobel results:', tmp)

        # resnext_laplace_results = process_search_results(resnext_embedding, resnext_800_laplace_database,
        #                                                  resnext_800_laplace_indices_mapping, 
        #                                                  sort_reverse=True)
        # resnext_sum = sum([d**2 for _, d in resnext_laplace_results])
        # resnext_laplace_results = [(chapter_id, similarity**2/resnext_sum) for chapter_id, similarity in resnext_laplace_results]
        # tmp = resnext_laplace_results[:top_n//5]
        # s = sum([d**2 for _, d in tmp])
        # tmp = [(chapter_id, distance**2/s) for chapter_id, distance in tmp]
        # print('ResNeXt 800 Laplace results:', tmp)
        
        img = vit_preprocess(template).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = vit_model(img).squeeze().unsqueeze(0).cpu().numpy()
        vit_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        vit_results = process_search_results(vit_embedding, vit_800_embed_database, 
                                             vit_800_embed_indices_mapping, sort_reverse=False)
        vit_results = [(chapter_id, 1 / (1 + np.e ** (4 * distance - 2))) for chapter_id, distance in vit_results]
        vit_sum = sum([d**2 for _, d in vit_results])
        vit_results = [(chapter_id, distance**2/vit_sum) for chapter_id, distance in vit_results]
        tmp = vit_results[:top_n]
        s = sum([d for _, d in tmp])
        tmp = [(chapter_id, distance/s) for chapter_id, distance in tmp]
        print('ViT 800 results:', tmp)

    final_results = dict()
    if resnext_embed_results is not None:
        for chapter_id, resnext_score in resnext_embed_results:
            final_results[chapter_id] = resnext_score / 2

    if resnext_sobel_results is not None:
        for chapter_id, resnext_score in resnext_sobel_results:
            if chapter_id in final_results:
                final_results[chapter_id] += resnext_score / 3
            else:
                final_results[chapter_id] = resnext_score / 3

    if resnext_laplace_results is not None:
        for chapter_id, resnext_score in resnext_laplace_results:
            if chapter_id in final_results:
                final_results[chapter_id] += resnext_score / 4
            else:
                final_results[chapter_id] = resnext_score / 4

    if vit_results is not None:
        for chapter_id, vit_score in vit_results:
          if chapter_id in final_results:
              final_results[chapter_id] += vit_score
          else:
              final_results[chapter_id] = vit_score

    final_results = [(chapter_id, final_results[chapter_id]) for chapter_id in final_results.keys()]
    final_results.sort(key=lambda x: x[1], reverse=True)

    final_results = final_results[:top_n]

    scores = [score for _, score in final_results]
    chapter_ids = [id for id, _ in final_results]

    scores = scores[:top_n]
    scores = scores / np.sum(scores)
    chapter_ids = chapter_ids[:top_n]
    
    response = []
    for i in range(top_n):
        response.append((chapter_ids[i], scores[i], get_message(chapter_ids[i])))

    return response
    
async def downloader(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Download file
    new_file = await update.message.effective_attachment[-1].get_file()
    file = await new_file.download_as_bytearray()
    
    return file

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = '''
HAPPY LUNAR NEW YEAR - from Esh
Welcome, I can help you searching YouShouYan (有兽焉) chapters. Please chat with me in English.
'''

    await update.message.reply_text(text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = '''
You can control me by sending these commands:
/start - Start the bot
/help - Get help about using the bot

/search -ch <text> <image> [-top <number>] [-sortChapterId]
Searching top relative chapters by text and/or image, text should place between brackets like "text", 'text' or `text`, currently accept multiple keywords and single image, command in [] is optional. Example:
\t/search -ch -text " DiTiNg|=) BirthDAY. Pikachu" <Image> -top 10 -sortChapterId
'''
    await update.message.reply_text(text)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.text != None:
            print(update.message.text)
            args = context.args
        else:
            print(update.message.caption)
            args = update.message.caption.split(' ')[1:]

        sortChapterId = '-sortChapterId' in ' '.join(args)

        if args[0].lower() in ('-chapter', '-ch'):
            if '-top' not in ' '.join(args).lower():
                args.append('-top')
                args.append('5')

            if '\"' not in ' '.join(args).lower():
                args.insert(1, None)

            topN = int(args[-1])

            if args[1] is not None:
                search_text = ' '.join(args[1:])[:' '.join(args[1:]).find('-top')].strip('\',"`').lower()
                # search_text = remove_punctuation(search_text)
                response_text = search_chapter_by_text(search_text, top_n=topN*5)
            else:
                response_text = None

            if (update.message and update.effective_chat and update.message.photo):
                file = await downloader(update, context)
                
                if not file:
                    await update.message.reply_text("Something went wrong, try again")
                    return
                
                template = Image.open(io.BytesIO(file)).convert("RGB")

                response_image = search_chapter_by_image(template, top_n=topN*5)
            else:
                response_image = None

            tmp = dict()
            if response_text is not None:
                for chapter_id, score, message_id in response_text:
                    tmp[chapter_id] = (score, message_id)

            if response_image is not None:
                for chapter_id, score, message_id in response_image:
                    if chapter_id not in tmp.keys():
                        tmp[chapter_id] = (score, message_id)
                    else:
                        tmp[chapter_id] = (tmp[chapter_id][0] + score, message_id)
            
            response = [(chapter_id, tmp[chapter_id][0], tmp[chapter_id][1]) for chapter_id in tmp.keys()]
            response.sort(key=lambda x: x[1], reverse=True)
            response = response[:topN]
            sum_score = sum([s for _, s, _ in response])
            response = [(chapter_id, score/sum_score, message_id) for chapter_id, score, message_id in response]
            if sortChapterId:
                response.sort(key=lambda x: x[0])

            print(response)
            response = [f"[Chapter {chapter_id}](https://t.me/youshouyan/{message_id}) Score: {score:.2f}" for chapter_id, score, message_id in response]
            
            await update.message.reply_text(
                'The chapters you are searching for\n' + 
                '\n'.join(response),
                parse_mode="Markdown"
            )
        else:
            print(args)
            await update.message.reply_text('Please type something to search')
    except Exception as e:
        print(e)
        await update.message.reply_text('Error, please check the message syntax')

# Responses
def handle_response(text: str) -> str:
    tmp = text.lower()

    return 'This is response'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User {update.message.chat.id} in {message_type}: {text}')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)
    
    print(f'Bot:', {response})
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} cause error {context.error}')

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('search', search_command))

    # Messages
    # app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, search_command))

    # Errors
    app.add_error_handler(error)

    # Run
    print('Starting...')
    app.run_polling(poll_interval=3)