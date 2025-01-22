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

# ========== ResNeXt model ==========
with open(YSY_COMIC_800_EMBED_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_800_embed_resnext = json.load(f)

resnext_800_indices_mapping = [chapter_800_embed_resnext[i]['file'][13:17] for i in range(len(chapter_800_embed_resnext))]
embeddings = np.array([chapter_800_embed_resnext[i]['embedding'] for i in range(len(chapter_800_embed_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_800_database = faiss.IndexFlatIP(2048)
resnext_800_database.add(embeddings)
print('Created database ResNeXt with patch size 800')
del chapter_800_embed_resnext, embeddings

with open(YSY_COMIC_400_EMBED_RESNEXT, 'r', encoding="utf-8") as f:
    chapter_400_embed_resnext = json.load(f)

resnext_400_indices_mapping = [chapter_400_embed_resnext[i]['file'][13:17] for i in range(len(chapter_400_embed_resnext))]
embeddings = np.array([chapter_400_embed_resnext[i]['embedding'] for i in range(len(chapter_400_embed_resnext))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
resnext_400_database = faiss.IndexFlatIP(2048)
resnext_400_database.add(embeddings)
print('Created database ResNeXt with patch size 400')
del chapter_400_embed_resnext, embeddings

resnext_model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
resnext_model = torch.nn.Sequential(*list(resnext_model.children())[:-1])  # Remove final layer
resnext_model = resnext_model.to(device)
resnext_model.eval()
resnext_preprocess = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2.transforms()
print('Loaded ResNeXt model')
    
# Vision Transformer
with open(YSY_COMIC_800_EMBED_VIT, 'r', encoding="utf-8") as f:
    chapter_800_embed_vit = json.load(f)

vit_800_indices_mapping = [chapter_800_embed_vit[i]['file'][13:17] for i in range(len(chapter_800_embed_vit))]
embeddings = np.array([chapter_800_embed_vit[i]['embedding'] for i in range(len(chapter_800_embed_vit))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
vit_800_database = faiss.IndexFlatL2(1280)
vit_800_database.add(embeddings)
print('Created database ViT with patch size 800')
del chapter_800_embed_vit, embeddings

with open(YSY_COMIC_400_EMBED_VIT, 'r', encoding="utf-8") as f:
    chapter_400_embed_vit = json.load(f)

vit_400_indices_mapping = [chapter_400_embed_vit[i]['file'][13:17] for i in range(len(chapter_400_embed_vit))]
embeddings = np.array([chapter_400_embed_vit[i]['embedding'] for i in range(len(chapter_400_embed_vit))])
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
vit_400_database = faiss.IndexFlatL2(1280)
vit_400_database.add(embeddings)
print('Created database ViT with patch size 400')
del chapter_400_embed_vit, embeddings

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
    # Matching score with OCR
    results = [(chapter[:4], get_text_matching_score(text, chapter)) for chapter in chapter_text.split('='*40+'\n') if chapter[:4].isnumeric()]
    
    # Matching score by semantic
    # for i in range(len(results)):
    #     chapter_id = results[i][0]
    #     pass

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

def process_search_results(image, model, preprocess, database, indices_mapping, sort_reverse=False):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img).squeeze().unsqueeze(0).cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

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

# def sliding_window_matching(image_path, folder_path, template, stride=None):
#     image = Image.open(os.path.join(folder_path, image_path)).convert("RGB")
#     chapter_id = image_path[13:17]
#     h, w = template.size
#     if stride is None:
#         avg_size = (h + w) / 2
#         stride = int(avg_size / 2)

#     max_score = -1.0
#     for y in range(0, image.size[1] - h + 1, stride):
#         for x in range(0, image.size[0] - w + 1, stride):
#             if y + h <= image.size[1] and x + w <= image.size[0]:
#                 roi = image.crop((y, x, y + h, x + w))

#                 roi_array = np.array(roi).flatten()
#                 template_array = np.array(template).flatten()

#                 # Normalize arrays
#                 roi_array = (roi_array - np.mean(roi_array)) / (np.std(roi_array) + 1e-6)
#                 template_array = (template_array - np.mean(template_array)) / (np.std(template_array) + 1e-6)

#                 # Compute correlation
#                 score = np.corrcoef(roi_array, template_array)[0, 1]
#                 if score > max_score:
#                     max_score = score ** 2
            
#     return (chapter_id, max_score)

# def parallel_template_matching(template, folder_path='database\You.Shou.Yan-comic-en'):
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda image_path: sliding_window_matching(image_path, folder_path, template), os.listdir(folder_path)[:200])
    return [chapter_id for chapter_id in results if chapter_id is not None]

def search_chapter_by_image(template, top_n=5):
    w, h = template.size
    if w * h < 500*500:
        resnext_results = process_search_results(template, resnext_model, resnext_preprocess, resnext_400_database,
                                                resnext_400_indices_mapping, sort_reverse=True)
        resnext_sum = sum([d for _, d in resnext_results])
        resnext_results = [(chapter_id, similarity/resnext_sum) for chapter_id, similarity in resnext_results]
        tmp = resnext_results[:top_n]
        s = sum([d for _, d in tmp])
        tmp = [(chapter_id, distance/s) for chapter_id, distance in tmp]
        print('ResNeXt 400 results:', tmp)

        vit_results = process_search_results(template, vit_model, vit_preprocess, vit_400_database,
                                            vit_400_indices_mapping, sort_reverse=False)
        vit_results = [(chapter_id, 1 / (1 + np.e ** (4 * distance - 2))) for chapter_id, distance in vit_results]
        vit_sum = sum([d for _, d in vit_results])
        vit_results = [(chapter_id, distance/vit_sum) for chapter_id, distance in vit_results]
        tmp = vit_results[:top_n]
        s = sum([d for _, d in tmp])
        tmp = [(chapter_id, distance/s) for chapter_id, distance in tmp]
        print('ViT 400 results:', tmp)
    else:
        resnext_results = process_search_results(template, resnext_model, resnext_preprocess, resnext_800_database,
                                                resnext_800_indices_mapping, sort_reverse=True)
        resnext_sum = sum([d for _, d in resnext_results])
        resnext_results = [(chapter_id, similarity/resnext_sum) for chapter_id, similarity in resnext_results]
        tmp = resnext_results[:top_n]
        s = sum([d for _, d in tmp])
        tmp = [(chapter_id, distance/s) for chapter_id, distance in tmp]
        print('ResNeXt 800 results:', tmp)
    
        vit_results = process_search_results(template, vit_model, vit_preprocess, vit_800_database,
                                            vit_800_indices_mapping, sort_reverse=False)
        vit_results = [(chapter_id, 1 / (1 + np.e ** (4 * distance - 2))) for chapter_id, distance in vit_results]
        vit_sum = sum([d for _, d in vit_results])
        vit_results = [(chapter_id, distance/vit_sum) for chapter_id, distance in vit_results]
        tmp = vit_results[:top_n]
        s = sum([d for _, d in tmp])
        tmp = [(chapter_id, distance/s) for chapter_id, distance in tmp]
        print('ViT 800 results:', tmp)

    final_results = dict()
    for chapter_id, resnext_score in resnext_results:
        final_results[chapter_id] = resnext_score / 2

    for chapter_id, vit_score in vit_results:
        if chapter_id in final_results:
            final_results[chapter_id] = final_results[chapter_id] + vit_score / 2
        else:
            final_results[chapter_id] = vit_score / 2

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
Welcome, I can answer some questions about YouShouYan on both manhua and donghua. Please chat with me in English.
'''

    await update.message.reply_text(text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = '''
You can control me by sending these commands:
/start - Start the bot
/help - Get help about using the bot

/search [-chapter/-Ch] [-text/-T] <text>
Searching chapter by text, currently only process single keywords. Example:
\t/search -chapter -text DiTiNg
\t/search -Ch -T Tuye

/search [-chapter/-Ch] [-image/-I] <image>
Searching chapter by image, currently only process single images.
'''
    await update.message.reply_text(text)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text != None:
        print(update.message.text)
        args = context.args
    else:
        print(update.message.caption)
        args = update.message.caption.split(' ')[1:]

    if args[0] in ('-chapter', '-Ch'):
        if args[1] in ('-text', '-T'):
            args[2] = ' '.join(args[2:])
            search_text = args[2].strip('\',"`').lower()
            
            response = search_chapter_by_text(search_text)

            print(response)
            response = [f"[Chapter {chapter_id}](https://t.me/youshouyan/{message_id}) Score: {score:.2f}" for chapter_id, score, message_id in response]
            
            await update.message.reply_text(
                'The chapters you are searching for\n' + 
                '\n'.join(response),
                parse_mode="Markdown"
            )
        elif args[1] in ('-image', '-I'):
            if (update.message and update.effective_chat and update.message.photo):
                file = await downloader(update, context)
                
                if not file:
                    await update.message.reply_text("Something went wrong, try again")
                    return
                
                template = Image.open(io.BytesIO(file)).convert("RGB")

                response = search_chapter_by_image(template)
                print(response)

                response = [f"[Chapter {chapter_id}](https://t.me/youshouyan/{message_id}) Score: {score:.2f}" for chapter_id, score, message_id in response]
                await update.message.reply_text(
                    'The chapters you are searching for\n' + 
                    '\n'.join(response),
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text('Please provide a photo to search')

            pass
    elif args[0] in ('-episode', '-Ep'):
        pass

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