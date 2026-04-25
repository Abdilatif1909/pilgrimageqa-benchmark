import json
import random
import hashlib
from pathlib import Path
from collections import Counter

random.seed(42)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FULL_DATASET = DATA_DIR / "pilgrimage_dataset.json"
TRAIN_DATASET = DATA_DIR / "pilgrimage_train.json"
DEV_DATASET = DATA_DIR / "pilgrimage_dev.json"
TEST_DATASET = DATA_DIR / "pilgrimage_test.json"

DATASET_TARGET_SIZE = 10240
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

SCIENTIFIC_INTENT_CATEGORIES = [
    "factual_place_lookup","route_navigation","nearest_entity_search","accommodation_guidance",
    "food_service_guidance","ritual_instruction","emergency_medical_support","emergency_security_support",
    "prayer_support","transport_advice","historical_information","cost_information","weather_time_advice",
    "crowd_movement_support","shopping_guidance","lost_person_help","long_context_reasoning",
    "translation_sensitive_query","ambiguous_query","multi_hop_query","visa_help","money_exchange",
    "sim_card_help","wheelchair_support","women_guidance","elderly_guidance","child_lost_help",
    "pharmacy_search","zamzam_collection","qurbani_guidance","ihram_issue","bathroom_search",
    "ablution_place","prayer_timing","language_translation_help"
]

REASONING_TYPES = ["single-hop", "multi-hop", "ambiguous", "translation-heavy"]
LINGUISTIC_FORMS = ["formal", "spoken", "colloquial", "noisy", "elliptical"]

DOMAIN_ENTITY_POOLS = {
    "core": [
        "Kaaba","Masjid al-Haram","Masjid an-Nabawi","Safa","Marwa","Mina","Arafat",
        "Muzdalifah","Jamarat","Clock Tower","Uhud","Quba Mosque","Haram Train",
        "Swissotel Makkah","Al Baik","Pilgrim Security Office","Makkah Emergency Hospital","Zamzam Station"
    ]
}

PARAPHRASE_CLUSTERS = {
    "factual_place_lookup": ["{place} qayerda joylashgan?","{place} qaysi hududda?"],
    "route_navigation": ["{place}ga qanday borsam bo'ladi?","Men {start}daman {place}ga yo'l ko'rsating"],
    "nearest_entity_search": ["{place}ga eng yaqin joy qaysi?","{place} yonida nima bor?"],
    "accommodation_guidance": ["{place}ga yaqin mehmonxona qaysi?","{place} atrofida tunash uchun joy kerak"],
    "food_service_guidance": ["{place} yonida ovqatlanadigan joy bormi?","{place} atrofida taom topsam bo'ladimi"],
    "ritual_instruction": ["{place}da qanday ibodat qilinadi?","{place}da nima qilish kerak?"],
    "emergency_medical_support": ["{place} atrofida tibbiy yordam bormi?","Agar betob bo'lsam {place}ga yaqin shifoxona qayerda?"],
    "emergency_security_support": ["{place}da xavfsizlik bo'limi qayerda?","Yo'qolib qolsam {place}da kimga murojaat qilaman?"],
    "prayer_support": ["{place}da namoz qayerda o'qiladi?","{place} yaqinida ibodat joyi bormi?"],
    "transport_advice": ["{place}dan transport topsam bo'ladimi?","{place}dan avtobus bormi?"],
    "historical_information": ["{place}ning tarixi qanday?","{place} nima uchun mashhur?"],
    "cost_information": ["{place} atrofida narxlar qimmatmi?","{place} yaqinidagi xizmatlar narxi qanday?"],
    "weather_time_advice": ["{place}ga qachon borgan yaxshi?","{place}da havo issiqmi?"],
    "crowd_movement_support": ["{place}da odam ko'p bo'lsa qayerdan yuray?","{place}da tirbandlikdan qanday chiqaman?"],
    "shopping_guidance": ["{place} yaqinida bozor bormi?","{place} yonida suvenir oladigan joy qayer?"],
    "lost_person_help": ["{place}da adashib qoldim nima qilay?","Odam yo'qolsa {place}da qayerga aytamiz?"],
    "long_context_reasoning": ["Men oilam bilan {start}daman keyin {place}ga o'tmoqchimiz maslahat bering"],
    "translation_sensitive_query": ["{place} rus tilida qanday ataladi?","{place} nomini ruschasini ayting"],
    "ambiguous_query": ["{place} bo'yicha yordam kerak","{place}ga yaqin joy kerak"],
    "multi_hop_query": ["{start}dan {place}ga borib keyin mehmonxonaga qanday o'taman?"],
    "visa_help": ["Vizaga oid muammo bo'lsa {place}da kimga murojaat qilaman?","{place}da visa yordami bormi?"],
    "money_exchange": ["{place} yaqinida pul almashtirish bormi?","{place}da valyuta ayirboshlash qayerda?"],
    "sim_card_help": ["{place} atrofida sim karta oladigan joy qayer?","{place}da internet uchun sim topsam bo'ladimi?"],
    "wheelchair_support": ["{place}da nogironlar aravachasi bormi?","{place} yaqinida wheelchair xizmati bormi?"],
    "women_guidance": ["{place}da ayollar uchun qulayliklar bormi?","{place} hududida ayollar qayerdan yuradi?"],
    "elderly_guidance": ["{place}da keksalar uchun qulay yo'l bormi?","{place}ga qariyalar qanday oson boradi?"],
    "child_lost_help": ["Bola yo'qolsa {place}da nima qilish kerak?","{place}da yo'qolgan bola uchun qayerga boramiz?"],
    "pharmacy_search": ["{place} yaqinida dorixona bormi?","{place} atrofida dori topsam bo'ladimi?"],
    "zamzam_collection": ["{place}da zamzam suvini qayerdan olaman?","{place} yaqinida zamzam tarqatish joyi bormi?"],
    "qurbani_guidance": ["{place}da qurbani bo'yicha ma'lumot bormi?","{place} yaqinida qurbani xizmatlari qayerda?"],
    "ihram_issue": ["{place}da ihram bilan bog'liq muammo bo'lsa nima qilaman?","{place} yaqinida ihram bo'yicha yordam kerak"],
    "bathroom_search": ["{place} yaqinida hojatxona qayerda?","{place}da restroom topsam bo'ladimi?"],
    "ablution_place": ["{place} atrofida tahorat joyi qayerda?","{place} yaqinida tahorat olish mumkinmi?"],
    "prayer_timing": ["{place}da namoz vaqti qachon?","{place} hududida keyingi namoz vaqti nechida?"],
    "language_translation_help": ["{place}da tarjimon yordam bormi?","{place} yaqinida til biladigan yordamchi bormi?"]
}

def deterministic_id(index:int)->str:
    return f"Q{index:05d}"

def choose_intent_category(index):
    return SCIENTIFIC_INTENT_CATEGORIES[index % len(SCIENTIFIC_INTENT_CATEGORIES)]

def choose_reasoning_type(intent):
    if "multi" in intent: return "multi-hop"
    if "ambiguous" in intent: return "ambiguous"
    if "translation" in intent: return "translation-heavy"
    return random.choice(REASONING_TYPES)

def choose_linguistic_form():
    return random.choice(LINGUISTIC_FORMS)

def choose_domain_entity():
    return random.choice(DOMAIN_ENTITY_POOLS["core"])

def choose_start_entity():
    return random.choice(DOMAIN_ENTITY_POOLS["core"])

def apply_linguistic_noise(text, form):
    if form == "colloquial":
        text = text.replace("qayerda","qatta").replace("qanday","qanaqa")
    elif form == "noisy":
        text = text.replace("bo'ladi","boladi").replace("kerak","kere")
    elif form == "elliptical":
        text = text.replace("qanday","").replace("qayerda","")
    return " ".join(text.split())

def build_uzbek_question(intent, entity, form):
    template = random.choice(PARAPHRASE_CLUSTERS[intent])
    q = template.format(place=entity, start=choose_start_entity())
    return apply_linguistic_noise(q, form)

def build_russian_question(intent, entity):
    return f"Помогите по запросу {intent.replace('_',' ')} возле {entity}."

def build_gold_answer_uz(intent, entity):
    return f"{entity} hududida {intent.replace('_',' ')} bo'yicha ziyoratchilar uchun kerakli xizmat, amaliy tavsiya va yo'nalishlar mavjud."

def build_gold_answer_ru(intent, entity):
    return f"В районе {entity} доступны необходимые сервисы, практические советы и маршруты по теме {intent.replace('_',' ')}."

def build_context_uz(intent, entity):
    return f"{entity} ziyoratchilar gavjum tashrif buyuradigan hudud bo‘lib, bu yerda {intent.replace('_',' ')} bilan bog‘liq xizmatlar, transport va yordam punktlari mavjud."

def build_context_ru(intent, entity):
    return f"{entity} является оживленной паломнической зоной, где доступны сервисы, транспорт и пункты помощи по теме {intent.replace('_',' ')}."

def assign_difficulty(reasoning, form):
    score = 0
    if reasoning == "multi-hop": score += 2
    elif reasoning in ["translation-heavy","ambiguous"]: score += 1
    if form in ["noisy","elliptical"]: score += 1
    if score <=1: return "easy"
    elif score==2: return "medium"
    return "hard"

def build_paraphrase_cluster_id(intent):
    return hashlib.md5(intent.encode()).hexdigest()[:8]

def assemble_benchmark_record(index):
    intent = choose_intent_category(index)
    reasoning = choose_reasoning_type(intent)
    form = choose_linguistic_form()
    entity = choose_domain_entity()
    return {
        "id": deterministic_id(index),
        "intent_category": intent,
        "reasoning_type": reasoning,
        "domain_entity": entity,
        "question_uz": build_uzbek_question(intent, entity, form),
        "question_ru": build_russian_question(intent, entity),
        "gold_answer_uz": build_gold_answer_uz(intent, entity),
        "gold_answer_ru": build_gold_answer_ru(intent, entity),
        "context_uz": build_context_uz(intent, entity),
        "context_ru": build_context_ru(intent, entity),
        "difficulty": assign_difficulty(reasoning, form),
        "linguistic_form": form,
        "paraphrase_cluster": build_paraphrase_cluster_id(intent)
    }

def build_full_benchmark_dataset(target_size=DATASET_TARGET_SIZE):
    return [assemble_benchmark_record(i) for i in range(1, target_size+1)]

def split_dataset(dataset):
    random.shuffle(dataset)
    n = len(dataset)
    train_end = int(n*TRAIN_RATIO)
    dev_end = train_end + int(n*DEV_RATIO)
    return dataset[:train_end], dataset[train_end:dev_end], dataset[dev_end:]

def save_json(data, path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def print_statistics(dataset):
    print("TOTAL RECORDS:", len(dataset))
    print("INTENT DISTRIBUTION:", Counter(x["intent_category"] for x in dataset))
    print("DIFFICULTY DISTRIBUTION:", Counter(x["difficulty"] for x in dataset))
    print("LINGUISTIC FORMS:", Counter(x["linguistic_form"] for x in dataset))

if __name__ == "__main__":
    dataset = build_full_benchmark_dataset()
    train, dev, test = split_dataset(dataset)
    save_json(dataset, FULL_DATASET)
    save_json(train, TRAIN_DATASET)
    save_json(dev, DEV_DATASET)
    save_json(test, TEST_DATASET)
    print_statistics(dataset)
    print("Dataset files saved into /data/")

