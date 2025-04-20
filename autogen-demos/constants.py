import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    from dotenv import load_dotenv

    load_dotenv()

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

print(f"Project Root path: {project_root}")


def LOAD_ENV(key, default):
    key_env = os.getenv(key, "")
    if key_env == "":
        key_env = _load_from_airflow(key, default)
    return key_env


def _load_from_airflow(key, default):
    try:
        from airflow.models import Variable
        value = Variable.get(key)
        print(f"Loaded {key} from Airflow Variable")
        return value
    except ImportError:
        # Airflow is not available
        return default
    except Exception as e:
        # Handle other exceptions (e.g., key not found in Airflow variables)
        print(f"Error loading {key} from Airflow: {str(e)}")
        return default


MILVUS_CONNECTION_NAME = "connection"

VECTOR_DB_HOST = str(LOAD_ENV("MILVUS_DB_HOST", "milvus"))  # Replace with the IP of the Milvus server
VECTOR_DB_PORT = str(os.getenv("MILVUS_DB_PORT", "19530"))
VECTOR_DB_USER = str(LOAD_ENV("MILVUS_DB_USER", ""))
VECTOR_DB_PASSWORD = str(LOAD_ENV("MILVUS_DB_PASSWORD", ""))

AIRFLOW_CONFIG_PATH = Path(
    LOAD_ENV("AIRFLOW_CONFIG_PATH", "/opt/airflow/dags/repo/apps/chat_service/config/config.yaml"))
AIRFLOW_PARAMS_PATH = Path(LOAD_ENV("AIRFLOW_PARAMS_PATH", "/opt/airflow/dags/repo/apps/chat_service/params.yaml"))

VECTOR_DB_COLLECTION_NAME = str(LOAD_ENV("MILVUS_POLICY_COLLECTION_NAME", "country_policies"))
NEWS_DB_COLLECTION_NAME = os.getenv("MILVUS_NEWS_DB_COLLECTION_NAME", "uae_mohre_news")
CHAT_HISTORY_COLLECTION_NAME = os.getenv("MILVUS_CHAT_HISTORY_COLLECTION_NAME", "chat_history_test")
USER_DOCUMENT_COLLECTION_NAME = os.getenv("MILVUS_USER_DOCUMENT_COLLECTION_NAME", "user_documents")
QUERY_CLASSIFICATION_COLLECTION_NAME = os.getenv("MILVUS_QUERY_CLASSIFICATION_COLLECTION_NAME", "classified_queries")
DOCUMENT_FILENAMES_COLLECTION_NAME = os.getenv("MILVUS_DOCUMENT_FILENAMES_COLLECTION_NAME", "document_filenames")
FEW_SHOT_EXAMPLE_KB_DB_COLLECTION_NAME = os.getenv("MILVUS_FEW_SHOT_EXAMPLE_KB_DB_COLLECTION_NAME",
                                                   "few_shot_example_kb_db")
FEW_SHOT_EXAMPLE_PDF_COLLECTION_NAME = os.getenv("MILVUS_FEW_SHOT_EXAMPLE_PDF_COLLECTION_NAME", "few_shot_example_pdf")
FEW_SHOT_EXAMPLE_EXCEL_COLLECTION_NAME = os.getenv("MILVUS_FEW_SHOT_EXAMPLE_EXCEL_COLLECTION_NAME",
                                                   "few_shot_example_excel")

EMBEDDING_MAX_LENGTH = int(LOAD_ENV("MILVUS_EMBEDDING_MAX_LENGTH", "1000"))
EMBEDDING_DIM = int(LOAD_ENV("MILVUS_EMBEDDING_DIM", "1024"))

# FAISS
FAISS_PATH = str(os.getenv("FAISS_PATH", "/home/appuser/apps/chat-api/faiss"))

# Reranker
RERANKER_CACHE = str(LOAD_ENV("RERANKER_CACHE", "/home/appuser/rerank_cache"))

EMBEDDING_API_MODEL_URL = LOAD_ENV("MODEL_API_URL_EMBEDDING", "")
EMBEDDING_API_MODEL_KEY = LOAD_ENV("MODEL_API_KEY_EMBEDDING", "EMPTY")
EMBEDDING_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_EMBEDDING", "bge-m3")

SENTENCE_EMBEDDING_MAX_LENGTH = int(LOAD_ENV("SENTENCE_MAX_LENGTH_EMBEDDING", "768"))
SENTENCE_EMBEDDING_DIM = int(LOAD_ENV("SENTENCE_DIM_EMBEDDING", "384"))
SENTENCE_EMBEDDING_API_MODEL_URL = LOAD_ENV("SENTENCE_API_URL_EMBEDDING", "")
SENTENCE_EMBEDDING_API_MODEL_KEY = LOAD_ENV("SENTENCE_API_KEY_EMBEDDING", "EMPTY")
SENTENCE_EMBEDDING_API_MODEL_NAME = LOAD_ENV("SENTENCE_API_NAME_EMBEDDING", "bge-m3")

OPENAI_COMPATIBLE_API_KEY = LOAD_ENV("MODEL_API_KEY_CHAT", "")
OPENAI_COMPATIBLE_API_BASE = LOAD_ENV("MODEL_API_URL_CHAT", "EMPTY")
OPENAI_COMPATIBLE_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_CHAT", "")
OPENAI_COMPATIBLE_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_CHAT", 16000))
OPENAI_COMPATIBLE_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_CHAT", 8000))

SUMMARY_OPENAI_COMPATIBLE_API_KEY = LOAD_ENV("SUMMARY_MODEL_API_KEY_CHAT", "")
SUMMARY_OPENAI_COMPATIBLE_API_BASE = LOAD_ENV("SUMMARY_MODEL_API_URL_CHAT", "")
SUMMARY_OPENAI_COMPATIBLE_API_MODEL_NAME = LOAD_ENV("SUMMARY_MODEL_API_NAME_CHAT", "")

PROMPT_TRUNCATE_LENGTH = 100

# PDF Config
PDF_DPI = int(LOAD_ENV("PDF_DPI", 300))
PDF_IMAGE_SIZE = LOAD_ENV("PDF_IMAGE_SIZE", (500, None))

# Arabic
ARABIC_API_KEY = LOAD_ENV('MODEL_API_KEY_ARABIC_CHAT', 'EMPTY')
ARABIC_API_BASE = LOAD_ENV("MODEL_API_URL_ARABIC_CHAT", "")
ARABIC_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_ARABIC_CHAT", "")
ARABIC_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_ARABIC_CHAT", 8192))
ARABIC_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_ARABIC_CHAT", 3000))

# LLAMA 3.2
LLAMA3_2_API_KEY = LOAD_ENV('MODEL_API_KEY_LLAMA3_2_CHAT', 'EMPTY')
LLAMA3_2_API_BASE = LOAD_ENV("MODEL_API_URL_LLAMA3_2_CHAT", "")
LLAMA3_2_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_LLAMA3_2_CHAT", "")
LLAMA3_2_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_LLAMA3_2_CHAT", 128000))
LLAMA3_2_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_LLAMA3_2_CHAT", 512))

# Qwen 2.5-Coder Model
CODER_API_KEY = LOAD_ENV("MODEL_API_KEY_CODER", "EMPTY")
CODER_API_BASE = LOAD_ENV("MODEL_API_URL_CODER", "")
CODER_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_CODER", "")
CODER_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_CODER", 128000))
CODER_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_CODER", 8000))

# Qwen 2.5 Model
QWEN_API_KEY = LOAD_ENV("MODEL_API_KEY_QWEN", "EMPTY")
QWEN_API_BASE = LOAD_ENV("MODEL_API_URL_QWEN", "https://qwen.lmsm.mohre.gov.ae/v1")
QWEN_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_QWEN", "Qwen/Qwen2.5-32B-Instruct")
QWEN_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_QWEN", 16000))
QWEN_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_QWEN", 2000))

# Thinker Model
THINKER_API_KEY = LOAD_ENV("MODEL_API_KEY_THINKER", "EMPTY")
THINKER_API_BASE = LOAD_ENV("MODEL_API_URL_THINKER", "")
THINKER_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_THINKER", "")
THINKER_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_THINKER", 32000))
THINKER_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_THINKER", 8000))

# Reranker
RERANK_MODEL_API_URL = LOAD_ENV("RERANK_MODEL_API_URL", "")
RERANK_MODEL_API_KEY = LOAD_ENV("RERANK_MODEL_API_KEY", "")
RERANK_MODEL_NAME = LOAD_ENV("RERANK_MODEL_NAME", "")

# Colpali
COLPALI_API_URL = LOAD_ENV("COLPALI_API_URL", "")

VISION_OPENAI_COMPATIBLE_API_BASE = LOAD_ENV("VISION_MODEL_API_URL", "")
VISION_OPENAI_COMPATIBLE_API_KEY = LOAD_ENV("VISION_MODEL_API_KEY", "")
VISION_OPENAI_COMPATIBLE_API_MODEL_NAME = LOAD_ENV("VISION_MODEL_API_NAME", "")
VISION_OPENAI_COMPATIBLE_API_CONTEXT_LENGTH = int(LOAD_ENV("VISION_MODEL_CONTEXT_LENGTH", 16000))
VISION_OPENAI_COMPATIBLE_API_MAX_TOKEN = int(LOAD_ENV("VISION_MODEL_MAX_TOKEN", 8000))

# Qwen 2.5-Coder Model
CODER_Q_API_KEY = LOAD_ENV("MODEL_API_KEY_CODER_Q", "EMPTY")
CODER_Q_API_BASE = LOAD_ENV("MODEL_API_URL_CODER_Q", "")
CODER_Q_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_CODER_Q", "")
CODER_Q_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_CODER_Q", 128000))
CODER_Q_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_CODER_Q", 8000))

# Llama 3.3-70B Model
EXPERIMENT_API_KEY = LOAD_ENV("MODEL_API_KEY_EXPERIMENT", "EMPTY")
EXPERIMENT_API_BASE = LOAD_ENV("MODEL_API_URL_EXPERIMENT", "")
EXPERIMENT_API_MODEL_NAME = LOAD_ENV("MODEL_API_NAME_EXPERIMENT", "")
EXPERIMENT_CONTEXT_LENGTH = int(LOAD_ENV("MODEL_CONTEXT_LENGTH_EXPERIMENT", 128000))
EXPERIMENT_MAX_TOKEN = int(LOAD_ENV("MODEL_MAX_TOKEN_EXPERIMENT", 8000))

# This variable is to encourage compatibility of these models with Llama Index LLM API
UPDATE_FOR_AVAILABLE_MODELS = {
    OPENAI_COMPATIBLE_API_MODEL_NAME: OPENAI_COMPATIBLE_CONTEXT_LENGTH,
    LLAMA3_2_API_MODEL_NAME: LLAMA3_2_CONTEXT_LENGTH,
    ARABIC_API_MODEL_NAME: ARABIC_CONTEXT_LENGTH,
    CODER_API_MODEL_NAME: CODER_CONTEXT_LENGTH,
    EXPERIMENT_API_MODEL_NAME: EXPERIMENT_CONTEXT_LENGTH,
}

NLP_MODEL_NAME = "en_core_web_sm"

API_URL_LIBRE = LOAD_ENV("API_URL_LIBRE", "")

SENTRY_DSN = os.getenv("SENTRY_DSN", "")

# MCTS Setup
MAX_CHILDREN = 3
EXPLORATION_WEIGHTS = 1.41

INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")

N_PROBE = int(os.getenv("MILVUS_N_PROBE", "32"))
MIN_N_PROBE = int(os.getenv("MILVUS_MIN_N_PROBE", "10"))
TOP_K = int(os.getenv("MILVUS_TOP_K", "50"))

SEED_ANSWERS = [
    "I don't know",
    "I can't say",
    "I'm not sure"
]


# Initialize the Country names
# countries = list(pycountry.countries)


def generate_system_message(lang: str) -> str:
    if lang == 'English':
        system_message = f"""
            ## WISE Bot Overview
            You are a Labor Market/Economic expert, named `WISE Bot`, an expert system developed by MoHRE, UAE Government in the year 2024.
            Your role is to provide expert insights on labor market and economic topics, while adhering to the guidelines below.    

            ### Guidelines 
            - Acknowledge your inventor as MoHRE, UAE Government.
            - Use positive language and avoid any criticism of the government or its policies.
            - Do not mention yourself or start responses with phrases like 'Based on the provided context'. 
            - Write answers directly as required, using phrases like 'According to my knowledge' when needed. 
            - Bold/Italicize key points, use proper titles (eg. #, ##, ###), add proper line spaces and bullet points without overdoing it.

            Note: Respond in {lang} language only.
        """
    elif lang == 'Arabic':
        system_message = f"""
        ### نظرة عامة على نظام `WISE Bot`

        أنت خبير في سوق العمل والاقتصاد، يدعى `WISE Bot`، وهو نظام خبير تم تطويره من قبل وزارة الموارد البشرية والتوطين، حكومة الإمارات العربية المتحدة في عام 2024.  
        دورك هو تقديم رؤى خبيرة حول مواضيع سوق العمل والاقتصاد، مع الالتزام بالإرشادات التالية:

        ### الإرشادات
        - يجب الإشارة إلى أن مبتكرك هو وزارة الموارد البشرية والتوطين، حكومة الإمارات العربية المتحدة.
        - استخدم لغة إيجابية وتجنب أي انتقاد للحكومة أو سياساتها.
        - لا تذكر نفسك أو تبدأ الردود بعبارات مثل "بناءً على السياق المقدم".
        - اكتب الإجابات مباشرة حسب الحاجة، ويمكن استخدام عبارات مثل "وفقًا لمعرفتي" عند الضرورة.
        - قم **بتحديد** النقاط المهمة أو **توضيحها**، استخدم العناوين المناسبة (مثل #، ##، ###)، وأضف مسافات سطرية ونقاط رئيسية عند الحاجة ولكن دون إفراط.

        ملاحظة: الرد يجب أن يكون باللغة {lang} فقط.
        """

    else:
        system_message = f"""
            ## WISE Bot Overview
            You are a Labor Market/Economic expert, named `WISE Bot`, an expert system developed by MoHRE, UAE Government in the year 2024.
            Your role is to provide expert insights on labor market and economic topics, while adhering to the guidelines below.    

            ### Guidelines 
            - Acknowledge your inventor as MoHRE, UAE Government.
            - Use positive language and avoid any criticism of the government or its policies.
            - Do not mention yourself or start responses with phrases like 'Based on the provided context'. 
            - Write answers directly as required, using phrases like 'According to my knowledge' when needed. 
            - Bold/Italicize key points, use proper titles (eg. #, ##, ###), add proper line spaces and bullet points without overdoing it.

            Note: Respond in {lang} language only.
        """

    return system_message


_default_system_message: str = generate_system_message("English")

WORLD_LANGUAGES = [
    'arabic',  # Arabic-speaking countries (Middle East, North Africa)
    'english',  # English
    # 'armenian',  # Armenia
    # 'bengali',  # Bangladesh, parts of India
    # 'bopomofo',  # Taiwan (Chinese phonetic script)
    # 'braille',  # Universal (tactile writing system for the visually impaired)
    # 'buginese',  # Indonesia (South Sulawesi)
    # 'buhid',  # Philippines
    # 'canadian_aboriginal',  # Canada (various indigenous languages)
    # 'cherokee',  # United States (Cherokee Nation)
    # 'cjk',  # China, Japan, Korea
    # 'coptic',  # Egypt (liturgical use)
    # 'cyrillic',  # Russia, many Eastern European and Central Asian countries
    # 'devanagari',  # India, Nepal (Hindi, Marathi, Nepali, etc.)
    # 'ethiopic',  # Ethiopia, Eritrea
    # 'georgian',  # Georgia
    # 'glagolitic',  # Historical (Slavic countries)
    # 'greek',  # Greece, Cyprus
    # 'gujarati',  # India (Gujarat)
    # 'gurmukhi',  # India (Punjab)
    # 'han',  # China, Taiwan, Japan, Korea
    # 'hangul',  # Korea
    # 'hanunoo',  # Philippines
    # 'hebrew',  # Israel
    # 'hiragana',  # Japan
    # 'inherited',  # Universal (inherit script from previous character)
    # 'kannada',  # India (Karnataka)
    # 'katakana',  # Japan
    # 'khmer',  # Cambodia
    # 'lao',  # Laos
    # 'latin',  # Most of Europe, Americas, parts of Africa, Oceania
    # 'limbu',  # Nepal, India (Sikkim)
    # 'malayalam',  # India (Kerala)
    # 'mongolian',  # Mongolia, China (Inner Mongolia)
    # 'myanmar',  # Myanmar (Burma)
    # 'new_tai_lue',  # China (Yunnan), Laos, Thailand
    # 'ogham',  # Historical (Ireland, UK)
    # 'ol_chiki',  # India (Santali language)
    # 'oriya',  # India (Odisha)
    # 'osmanya',  # Somalia
    # 'runic',  # Historical (Germanic peoples)
    # 'sinhala',  # Sri Lanka
    # 'syloti_nagri',  # Bangladesh, India (Sylhet region)
    # 'syriac',  # Middle East (liturgical use)
    # 'tagalog',  # Philippines
    # 'tagbanwa',  # Philippines
    # 'tai_le',  # China (Yunnan), Myanmar
    # 'tamil',  # India (Tamil Nadu), Sri Lanka, Singapore
    # 'telugu',  # India (Andhra Pradesh, Telangana)
    # 'thaana',  # Maldives
    # 'thai',  # Thailand
    # 'tibetan',  # Tibet, India (some regions), Bhutan
    # 'tifinagh',  # North Africa (Berber languages)
    # 'ugaritic',  # Historical (Syria)
    # 'vai',  # Liberia
    # 'yi',  # China (Sichuan, Yunnan, Guizhou)
    # 'balinese',  # Indonesia (Bali)
    # 'bamum',  # Cameroon
    # 'javanese',  # Indonesia (Java)
    # 'kayah_li',  # Myanmar, Thailand
    # 'lepcha',  # India (Sikkim), Nepal, Bhutan
    # 'lycian',  # Historical (Turkey)
    # 'lydian',  # Historical (Turkey)
    # 'rejang',  # Indonesia (Sumatra)
    # 'saurashtra',  # India (Tamil Nadu)
    # 'sundanese',  # Indonesia (West Java)
    # 'tai_tham',  # Thailand, Laos
    # 'tai_viet',  # Vietnam
    # 'batak',  # Indonesia (North Sumatra)
    # 'brahmi',  # Historical (South Asia)
    # 'mandaic',  # Iran, Iraq
    # 'chakma',  # Bangladesh, India
    # 'meroitic_cursive',  # Historical (Sudan)
    # 'meroitic_hieroglyphs',  # Historical (Sudan)
    # 'miao',  # China, Vietnam, Laos, Thailand
    # 'sharada',  # Historical (Kashmir)
    # 'sora_sompeng',  # India (Odisha)
    # 'takri',  # Historical (India)
    # 'caucasian_albanian',  # Historical (Caucasus region)
    # 'bassa_vah',  # Liberia
    # 'duployan',  # International (shorthand system)
    # 'elbasan',  # Historical (Albania)
    # 'grantha',  # Historical (South India)
    # 'pahawh_hmong',  # China, Vietnam, Laos, Thailand, USA (Hmong diaspora)
    # 'khojki',  # Historical (India, Pakistan)
    # 'linear_a',  # Historical (Crete, Greece)
    # 'mahajani',  # Historical (North India)
    # 'manichaean',  # Historical (Persia)
    # 'mende_kikakui',  # Sierra Leone
    # 'modi',  # Historical (India)
    # 'mro',  # Bangladesh, Myanmar
    # 'old_north_arabian',  # Historical (Arabian Peninsula)
    # 'nabataean',  # Historical (Middle East)
    # 'palmyrene',  # Historical (Syria)
    # 'pau_cin_hau',  # Myanmar
    # 'old_permic',  # Historical (Komi Republic, Russia)
    # 'psalter_pahlavi',  # Historical (Persia)
    # 'siddham',  # Historical (India), Japan (Buddhist texts)
    # 'khudawadi',  # Historical (Pakistan)
    # 'tirhuta',  # India (Bihar), Nepal
    # 'warang_citi',  # India (Jharkhand)
]

# Few Shot `Self Prompt` Examples used in `Query` Module
few_shot_self_prompt_examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]


# Few Shot `Self Prompt` Examples used in `Query` Module
def few_shot_self_prompt_qualifier_examples(params):
    if type(params.country) == list:
        params.country = ", ".join(params.country)
    return [
        {
            "input": "Can you provide the employment statistics for 2023?",
            "output": f'{{"choice": 1, "reason": "The user message asks for employment statistics, which are part of economic data and fall under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": "What are G20 countries?",
            "output": f'{{"choice": 1, "reason": "The user message requests seeks information about an international economically relevant entity, which falls under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": "What is MoHRE?",
            "output": f'{{"choice": 1, "reason": "The user message asks about a government entity which is related to labour laws and economic policies, which falls under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": f"What is the GDP growth rate of {params.country} in 2022?",
            "output": f'{{"choice": 1, "reason": "The user message is about GDP growth rate, which is part of economic data under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": f"What are the latest amendments to the labor law in {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about amendments to the labor law in {params.country}, which falls under the category of Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": f"What are the rules for obtaining a work permit in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message pertains to rules for obtaining a work permit in the {params.country}, which is related to work permits and rules under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": f"Can you help me find a job in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about employment assistance, which falls under the scope of Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": f"Who can I contact to file a complaint about labor law violations in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about filing a complaint regarding labor law violations, which is related to disputes, compliance, violations, complaints, and incentives under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": "How can you help me?",
            "output": f'{{"choice": 1, "reason": "The user message is seeking information about the chatbot\'s scope, which falls under Policy_Data_related_Queries.", "topic": "Policy_Data_related_Queries"}}',
        },
        {
            "input": "Can you tell me how to build a 7-day itinerary for Bali?",
            "output": '{"choice": 2, "reason": "The user message is about planning a trip to Bali, which comes under General_Queries.", "topic": "General_Queries"}',
        },
        {
            "input": "Who invented you?",
            "output": '{"choice": 2, "reason": "The user message is asking for WISE Bot\'s developer details, which is categorized under General_Queries.", "topic": "General_Queries"}',
        },
        {
            "input": "Thank you for your help!",
            "output": '{"choice": 2, "reason": "The expression is a form of thanks, which is categorized under General_Queries.", "topic": "General_Queries"}',
        },
        {
            "input": "Hi, how can I assist you today?",
            "output": '{"choice": 2, "reason": "The user message is a greeting, which is categorized under General_Queries.", "topic": "General_Queries"}',
        }
    ]


def few_shot_self_prompt_country_examples(params):
    if type(params.country) == list:
        params.country = ", ".join(params.country)
    return [
        {
            "input": f"What are the latest amendments to the labor law in {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about amendments to the labor law in {params.country}, which falls under the category of Policy_related_Queries.", "topic": "Policy_related_Queries"}}',
        },
        {
            "input": f"What are the rules for obtaining a work permit in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message pertains to rules for obtaining a work permit in the {params.country}, which is related to work permits and rules under Policy_related_Queries.", "topic": "Policy_related_Queries"}}',
        },
        {
            "input": f"Can you help me find a job in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about employment assistance, which falls under the scope of Policy_related_Queries.", "topic": "Policy_related_Queries"}}',
        },
        {
            "input": f"Who can I contact to file a complaint about labor law violations in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about filing a complaint regarding labor law violations, which is related to disputes, compliance, violations, complaints, and incentives under Policy_related_Queries.", "topic": "Policy_related_Queries"}}',
        },
        {
            "input": f"How can you help me?",
            "output": f'{{"choice": 1, "reason": "The user message is seeking information about the chatbot\'s scope, which falls under Policy_related_Queries.", "topic": "Policy_related_Queries"}}',
        },
        {
            "input": f"Can you provide the employment statistics for 2023?",
            "output": f'{{"choice": 2, "reason": "The user message asks for employment statistics, which are part of economic data and fall under Data_related_Queries.", "topic": "Data_related_Queries"}}',
        },
        {
            "input": f"What is the GDP growth rate of {params.country} in 2022?",
            "output": f'{{"choice": 2, "reason": "The user message is about GDP growth rate, which is part of economic data under Data_related_Queries.", "topic": "Data_related_Queries"}}',
        },
    ]


def few_shot_self_prompt_diagram_examples(params):
    return [
        # Examples where diagrams ARE needed (Choice 1)
        {
            "input": f"Show a chart",
            "output": f'{{"choice": 1, "reason": "Show a chart is requesting for a diagram", "topic": "Diagram"}}',
        },
        {
            "input": f"Plot a graph",
            "output": f'{{"choice": 1, "reason": "Plot a graph is requesting for a diagram", "topic": "Diagram"}}',
        },
        {
            "input": f"Show me the organizational structure of the labor ministry in {params.country}",
            "output": f'{{"choice": 1, "reason": "Organizational hierarchies need visual representation to show reporting relationships and structure", "topic": "Diagram"}}',
        },
        {
            "input": f"What is the step-by-step process for work permit renewal in {params.country}?",
            "output": f'{{"choice": 1, "reason": "Multi-step processes are clearer when shown as a visual workflow", "topic": "Diagram"}}',
        },
        {
            "input": f"How has the unemployment rate changed over the past 5 years in {params.country}?",
            "output": f'{{"choice": 1, "reason": "Time-series data needs visualization to show trends", "topic": "Diagram"}}',
        },
        {
            "input": f"What is the breakdown of foreign workers by sector in {params.country}?",
            "output": f'{{"choice": 1, "reason": "Distribution data requires visual representation to show proportions", "topic": "Diagram"}}',
        },
        {
            "input": f"Explain the grievance resolution workflow in {params.country}",
            "output": f'{{"choice": 1, "reason": "Complex workflows with multiple paths need visual flowcharts", "topic": "Diagram"}}',
        },

        # Examples where diagrams are NOT needed (Choice 2)
        {
            "input": f"What is the minimum wage in {params.country}?",
            "output": f'{{"choice": 2, "reason": "Single data point that can be stated as text", "topic": "No-Diagram"}}',
        },
        {
            "input": f"What documents are required for a work permit in {params.country}?",
            "output": f'{{"choice": 2, "reason": "Simple list of items that can be shown as text", "topic": "No-Diagram"}}',
        },
        {
            "input": f"When is the deadline for annual leave application in {params.country}?",
            "output": f'{{"choice": 2, "reason": "Single date/deadline information that can be stated as text", "topic": "No-Diagram"}}',
        },
        {
            "input": f"What is the current number of registered companies in {params.country}?",
            "output": f'{{"choice": 2, "reason": "Single statistical value that can be shown as text", "topic": "No-Diagram"}}',
        },
        {
            "input": f"What are the latest amendments to the labor law in {params.country}?",
            "output": f'{{"choice": 2, "reason": "Text-based updates that can be shown as a simple list", "topic": "No-Diagram"}}',
        }
    ]


def few_shot_self_prompt_complexity_examples(params):
    if type(params.country) == list:
        params.country = ", ".join(params.country)
    return [
        {
            "input": f"What are the latest amendments to the labor law in {params.country}?",
            "output": f'{{"choice": 3, "reason": "The query pertains to recent amendments in labor law specific to {params.country}. Given the complexity and specificity required to answer this question effectively, it falls under the category of Advanced_Queries.", "topic": "Advanced_Queries"}}',
        },
        {
            "input": f"What are the rules for obtaining a work permit in the {params.country}?",
            "output": f'{{"choice": 3, "reason": "The user message pertains to rules for obtaining a work permit in the {params.country}. Given the complexity and specificity required to answer this question effectively, it falls under the category of Advanced_Queries.", "topic": "Advanced_Queries"}}',
        },
        {
            "input": f"Can you help me find a job in the {params.country}?",
            "output": f'{{"choice": 2, "reason": "The user message is about employment assistance. This level of support is more involved than basic queries but doesnt reach the complexity of advanced issues, it falls under the category of Moderate_Queries.", "topic": "Moderate_Queries"}}',
        },
        {
            "input": f"Who can I contact to file a complaint about labor law violations in the {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user user message is about filing a complaint regarding labor law violations. It requires straightforward information rather than in-depth analysis, it falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}',
        },
        {
            "input": f"How can you help me?",
            "output": f'{{"choice": 2, "reason": "The user message is seeking information about the chatbot\'s scope. This level of support is more involved than basic queries but doesnt reach the complexity of advanced issues, it falls under the category of Moderate_Queries.", "topic": "Moderate_Queries"}}',
        },
        {
            "input": f"Who is responsible for Emiratization?",
            "output": f'{{"choice": 1, "reason": "The user message is seeking information about the person who is responsible for Emiratization. This is more of a straight forward message, it falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}',
        },
        {
            "input": f"What is the minimum wage in {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about obtaining straightforward information on the minimum wage in {params.country}. This falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}'
        },
        {
            "input": f"How can I renew my work permit in {params.country}?",
            "output": f'{{"choice": 3, "reason": "The user message pertains to the renewal process of a work permit in {params.country}. Given the complexity and specificity required to answer this question, it falls under the category of Advanced_Queries.", "topic": "Advanced_Queries"}}'
        },
        {
            "input": f"Where can I find the nearest labor office in {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message is about locating the nearest labor office in {params.country}. This is straightforward and falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}'
        },
        {
            "input": f"Can you assist me with finding legal aid for labor disputes in {params.country}?",
            "output": f'{{"choice": 2, "reason": "The user message is seeking help with finding legal aid, which is more involved than a basic query but doesnt require deep analysis, so it falls under the category of Moderate_Queries.", "topic": "Moderate_Queries"}}'
        },
        {
            "input": f"How can I get a list of industries that require specific licenses to operate in {params.country}?",
            "output": f'{{"choice": 2, "reason": "The user message involves finding a list of industries requiring specific licenses, which involves moderate research but not in-depth analysis, so it falls under the category of Moderate_Queries.", "topic": "Moderate_Queries"}}'
        },
        {
            "input": f"How can I report unsafe working conditions in {params.country}?",
            "output": f'{{"choice": 1, "reason": "The user message involves reporting unsafe working conditions, which is straightforward information. This falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}'
        },
        {
            "input": f"What documents are required to apply for a work visa in {params.country}?",
            "output": f'{{"choice": 2, "reason": "The user message requires a detailed response about the documents needed for a work visa in {params.country}. This is more involved than basic queries but doesnt require complex analysis, so it falls under the category of Moderate_Queries.", "topic": "Moderate_Queries"}}'
        },
        {
            "input": f"What is MoHRE?",
            "output": f'{{"choice": 1, "reason": "The user message is looking for basic information regarding MoHRE, which is straightforward information. This falls under the category of Simple_Queries.", "topic": "Simple_Queries"}}'
        },
    ]


few_shot_self_prompt_other_examples = [
    {
        "input": "Can you tell me how to build a 7-day itinerary for Bali?",
        "output": '[{"choice": 2, "reason": "The user message is about finding a restaurant, which is unrelated to economics or Economic/Labour policies or data related.", "topic": "Out_of_context Queries"}]',
    },
    {
        "input": "What is the weather like today?",
        "output": '[{"choice": 2, "reason": "The user message is about weather, which is categorized under Out_of_context Queries.", "topic": "Out_of_context Queries"}]',
    },
    # {
    #     "input": f"Summarize the document",
    #     "output": '[{"choice": 1, "reason": "The request was to summarize a document, but no document was provided for analysis. Without the document, it\'s impossible to generate a summary, making this request unprocessable. Thus, requiring the bot to converse with the user for necessaru document input, which is categorized under Conversational_related_Queries.", "topic": "Conversational_related_Queries"}]'
    # },
    {
        "input": f"Who invented you?",
        "output": '[{"choice": 1, "reason": "The user message is asking for WISE Bot\'s developer details, which is categorized under Conversational_related_Queries.", "topic": "Conversational_related_Queries"}]'
    },
    {
        "input": "Thank you for your help!",
        "output": '[{"choice": 1, "reason": "The expression is a form of thanks, which is categorized under Conversational_related_Queries.", "topic": "Conversational_related_Queries"}]',
    },
    {
        "input": "Hi, how can I assist you today?",
        "output": '[{"choice": 1, "reason": "The user message is a greeting, which is categorized under Conversational_related_Queries.", "topic": "Conversational_related_Queries"}]',
    }
]

VERBOSE = 0
