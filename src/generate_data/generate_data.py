# def generate_prompts_list(x, y, words, verb):
#     months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
#     month_to_num = {
#     'January': 'one', 'February': 'two', 'March': 'three', 'April': 'four', 'May': 'five', 'June': 'six',
#     'July': 'seven', 'August': 'eight', 'September': 'nine', 'October': 'ten', 'November': 'eleven', 'December': 'twelve'
#     }
#     months = [month_to_num[i] for i in months]
#     prompts_list = []
#     for j in range(1024): # this must come first else 1 2 3 overrepresented!
#         for i in range(x, y):
#             rand_words = random.sample(words, k=5)
#             prompt_dict = {
#                 'S1': months[i],
#                 'S2': months[i+1],
#                 'S3': months[i+2],
#                 'S4': months[i+3],
#                 'corr': f" {months[i+4]}",
#                 'incorr': f" {months[i+3]}",
#                 'text': f"{rand_words[0]} was {verb} in {months[i]}. {rand_words[1]} was {verb} in {months[i+1]}. {rand_words[2]} was {verb} in {months[i+2]}. {rand_words[3]} was {verb} in {months[i+3]}. {rand_words[4]} was {verb} in",
#             }
#             prompts_list.append(prompt_dict)
#     return prompts_list

# import copy
# month_to_num = {'one': 'January', 'two': 'February', 'three': 'March', 'four': 'April', 'five': 'May', 'six': 'June', 'seven': 'July', 'eight': 'August', 'nine': 'September', 'ten': 'October', 'eleven': 'November', 'twelve': 'December'}

# # Revised function to handle the AttributeError
# def replace_month_names(data_list):
#     out = copy.deepcopy(data_list)
#     for item in out:
#         # Replace month names in key-value pairs
#         for key in list(item.keys()):  # list() to avoid 'RuntimeError: dictionary changed size during iteration'
#             value = item[key]
#             if value in month_to_num:
#                 item[key] = month_to_num[value]
#             elif key == 'corr' or key == 'incorr':
#                 item[key] = " " + month_to_num[value.replace(" ", '')]

#         # Replace month names in text fields
#         if 'text' in item:
#             text = item['text']
#             for month_name, month_num in month_to_num.items():
#                 text = text.replace(month_name, str(month_num))
#             item['text'] = text

#     return out

# # nw template dataset, sold
# ###############
# import random

# # List of common, short words which are likely to be single tokens in GPT-2
# common_words = [
#     "Apple", "Ball", "Car", "Dog", "Egg", "Fish", "Gold", "Hat", "Ink", "Jar",
#     "Kite", "Lamp", "Moon", "Nest", "Owl", "Pig", "Quilt", "Rat", "Sun", "Tree",
#     "Umbrella", "Vase", "Wolf", "Yarn", "Zip", "Bird", "Cat", "Drum", "Frog",
#     "Grape", "House", "Ice", "Juice", "Key", "Leaf", "Map", "Nut", "Orange",
#     "Piano", "Queen", "Ring", "Star", "Train", "Van", "Whale", "Xylophone",
#     "Yacht", "Zebra", "Ax", "Box", "Cow", "Desk", "Ear", "Fan", "Gate", "Hill",
#     "Iron", "Joke", "King", "Lion", "Milk", "Nose", "Oil", "Pen", "Quiz", "Rose",
#     "Shoe", "Tail", "Vine", "Wall", "Year", "Ant", "Bug", "Corn", "Duck", "Fire",
#     "Grass", "Hand", "Island", "Jam", "Knee", "Lake", "Mouse", "Nail", "Pear",
#     "Quack", "Road", "Sand", "Tent", "Valley", "Wind", "Yard", "Arm", "Boat",
#     "Cake", "Door", "Eye", "Flag", "Horse", "Jeep", "Knife", "Light", "Mountain",
#     "Night", "Ocean", "Plate", "Queen", "Rain", "Snow", "Tree", "Umbrella",
#     "Valley", "Window", "Yogurt", "Zoo"
# ]
# random_single_word_objects = [obj.capitalize() for obj in common_words]

# def filter_to_single_token(words):
#     return [w for w in words if len(model.tokenizer.tokenize(w)) == 1]
# random_single_word_objects = filter_to_single_token(random_single_word_objects)
# len(random_single_word_objects)

# # nw template dataset, made
# ###############



# ###############
# def generate_prompts_list_corr(prompt_list):
#     outlist = []
#     for prompt_dict in prompts_list:
#         r1 = random.randint(1, 12)
#         r2 = random.randint(1, 12)
#         while True:
#             r3 = random.randint(1, 12)
#             r4 = random.randint(1, 12)
#             if r4 - 1 != r3:
#                 break
#         new_text = prompt_dict['text'].replace(prompt_dict['S1'], str(r1)).replace(prompt_dict['S2'], str(r2)).replace(prompt_dict['S3'], str(r3)).replace(prompt_dict['S4'], str(r4))
#         new_prompt_dict = {
#             'S1': str(r1),
#             'S2': str(r2),
#             'S3': str(r3),
#             'S4': str(r4),
#             'corr': prompt_dict['corr'],
#             'incorr': prompt_dict['incorr'],
#             'text': new_text
#         }
#         outlist.append(new_prompt_dict)
#     return outlist

# # prompts_list_2 = generate_prompts_list_corr(prompts_list)

# # import pickle
# # from google.colab import files

# # with open('randDS_numerals.pkl', 'wb') as file:
# #     pickle.dump(prompts_list_2, file)
# # files.download('randDS_numerals.pkl')