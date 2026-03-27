import json
import random
import hashlib
from pathlib import Path

FIRST_NAMES_M = [
    "Алмас", "Арман", "Бауыржан", "Дамир", "Ержан", "Куаныш", "Нурлан",
    "Санжар", "Тимур", "Ильяс", "Даулет", "Мирас", "Адиль", "Руслан",
    "Азамат", "Берик", "Данияр", "Ерлан", "Кайрат", "Марат", "Нуржан",
    "Ринат", "Серик", "Тахир", "Аскар", "Олжас", "Жандос", "Максат",
]

FIRST_NAMES_F = [
    "Айгерим", "Алия", "Дана", "Жанна", "Камила", "Мадина", "Назерке",
    "Сабина", "Томирис", "Фатима", "Аружан", "Гульнара", "Дарига",
    "Инжу", "Карлыгаш", "Лаура", "Меруерт", "Нургуль", "Раушан",
    "Салтанат", "Улжан", "Хадиша", "Динара", "Асель", "Балжан",
]

LAST_NAMES = [
    "Абдрахманов", "Бейсембаев", "Габдуллин", "Джумабеков", "Ермеков",
    "Жумабаев", "Кенжебаев", "Мусабаев", "Нурпеисов", "Оспанов",
    "Рахимов", "Сулейменов", "Тлеубердин", "Утегенов", "Хасенов",
    "Шарипов", "Алтынбеков", "Байтурсынов", "Валиханов", "Искаков",
    "Касымов", "Муканов", "Нургалиев", "Сагинтаев", "Токаев",
]

CITIES = [
    "Алматы", "Астана", "Шымкент", "Караганда", "Актобе", "Тараз",
    "Павлодар", "Усть-Каменогорск", "Семей", "Атырау", "Костанай",
    "Кызылорда", "Уральск", "Петропавловск", "Актау", "Талдыкорган",
]

UNIVERSITIES = [
    "Казахский национальный университет им. аль-Фараби",
    "Евразийский национальный университет им. Гумилёва",
    "Назарбаев Университет",
    "КИМЭП",
    "Казахстанско-Британский технический университет",
    "Алматы Менеджмент Университет",
    "Международный университет информационных технологий",
    "Satbayev University",
    "Университет КАЗГЮУ",
    "Suleyman Demirel University",
    "Astana IT University",
    "Казахский агротехнический университет",
]

SKILLS_POOL = [
    "Python", "JavaScript", "Data Science", "Machine Learning",
    "Project Management", "Leadership", "Public Speaking", "Analytics",
    "UX Design", "Product Management", "Marketing", "Finance",
    "Entrepreneurship", "Critical Thinking", "Research", "Statistics",
    "SQL", "Cloud Computing", "Design Thinking", "Communication",
    "Negotiation", "Strategic Planning", "Agile/Scrum", "Blockchain",
    "Digital Marketing", "Content Creation", "Video Editing",
    "Graphic Design", "Copywriting", "SEO", "Social Media",
    "Программирование", "Аналитика", "Управление проектами",
]

LANGUAGES_POOL = [
    "Казахский", "Русский", "Английский", "Турецкий", "Китайский",
    "Немецкий", "Французский", "Арабский", "Корейский", "Испанский",
]

ACHIEVEMENT_CATEGORIES = [
    "academic", "leadership", "technology", "sports", "volunteer",
    "arts", "entrepreneurship", "science", "community",
]

ACHIEVEMENT_TEMPLATES = {
    "academic": [
        "Государственная стипендия «Болашак»",
        "Диплом с отличием по специальности {field}",
        "Победитель олимпиады по {subject}",
        "Стипендия ректора за академические достижения",
        "Публикация в научном журнале по теме {topic}",
    ],
    "leadership": [
        "Президент студенческого совета",
        "Основатель клуба {club_name}",
        "Координатор волонтёрского движения",
        "Организатор TEDx мероприятия в университете",
        "Капитан университетской команды по дебатам",
    ],
    "technology": [
        "Победитель хакатона {hackathon_name}",
        "Разработчик мобильного приложения {app_name}",
        "Участник Google Summer of Code",
        "Сертификация AWS/GCP/Azure",
        "Контрибьютор в open-source проект",
    ],
    "sports": [
        "Чемпион города по {sport}",
        "Участник сборной Казахстана по {sport}",
        "Призёр универсиады",
    ],
    "volunteer": [
        "Волонтёр EXPO-2017",
        "Участник программы AIESEC",
        "Наставник для школьников из регионов",
        "Волонтёр в детском доме (2+ года)",
        "Организатор благотворительного марафона",
    ],
    "entrepreneurship": [
        "Основатель стартапа {startup_name}",
        "Победитель конкурса бизнес-планов",
        "Участник инкубатора Astana Hub",
        "Грант на развитие бизнес-проекта",
    ],
}

LEADERSHIP_ROLES_POOL = [
    "Президент студенческого совета",
    "Руководитель проектной команды",
    "Координатор волонтёрского движения",
    "Капитан спортивной команды",
    "Староста курса",
    "Организатор мероприятий",
    "Наставник (ментор) для младших студентов",
    "Основатель студенческого клуба",
    "Председатель научного общества",
    "Лидер дебатного клуба",
]

PROJECTS_POOL = [
    "Платформа для онлайн-обучения школьников",
    "Мобильное приложение для мониторинга экологии",
    "Система автоматизации учёта для МСБ",
    "Исследование рынка EdTech в Центральной Азии",
    "Чат-бот для студенческой поддержки",
    "Анализ данных городского транспорта",
    "Платформа для социального предпринимательства",
    "ML-модель для прогнозирования урожайности",
    "Веб-портал для НКО Казахстана",
    "VR-тур по историческим местам Казахстана",
    "IoT-система умного дома",
    "Блокчейн-решение для цепи поставок",
]

ESSAY_PROMPTS = [
    "Расскажите о вашей мотивации поступить в inVision U и как это связано с вашими жизненными целями.",
    "Опишите ситуацию, в которой вы проявили лидерские качества и чему это вас научило.",
    "Какую проблему в обществе вы хотели бы решить и почему?",
]

ESSAY_TEMPLATES_MOTIVATION = [
    """С самого детства я мечтал{a} создавать технологии, которые изменят жизнь людей. Родившись в {city}, я видел, как цифровизация постепенно трансформирует нашу страну, и это вдохновило меня выбрать путь в сфере {field}.

Мой путь в {university} научил меня не только техническим навыкам, но и критическому мышлению. Работая над проектом {project}, я осознал{a}, что настоящие инновации рождаются на стыке технологий и понимания потребностей людей.

inVision U для меня — это уникальная возможность объединить мою страсть к технологиям с желанием создавать значимые решения. Программа университета, основанная на принципах инновационного мышления, идеально совпадает с моими целями.

Я верю, что именно здесь я смогу развить свой потенциал и внести вклад в развитие технологической экосистемы Казахстана. Мой опыт волонтёрства и работы в команде подготовил меня к интенсивной программе inVision U.

В будущем я планирую {future_plan}. Я уверен{a}, что знания и связи, полученные в inVision U, станут фундаментом для реализации этих амбициозных планов.""",

    """Когда я впервые узнал{a} об inVision U, я понял{a}: это именно тот университет, который формирует лидеров нового поколения. Мой опыт в {field} и стремление к постоянному развитию привели меня к этой заявке.

За время учёбы в {university} я участвовал{a} в нескольких проектах, которые укрепили мою уверенность в выбранном пути. Особенно значимым стал проект {project}, где я научил{a}ся работать в условиях неопределённости и принимать решения под давлением.

Мотивация поступить в inVision U связана с моим глубоким убеждением, что образование должно быть практико-ориентированным. Я хочу не просто изучать теории, а создавать реальные продукты, которые решают реальные проблемы.

{city} — мой родной город, и я вижу огромный потенциал для развития технологических решений в нашем регионе. С дипломом inVision U я смогу вернуться и внести значимый вклад в его развитие.

Мой план — {future_plan}, и я готов{a} работать усердно, чтобы достичь этой цели.""",

    """Мой путь к inVision U начался с простого вопроса: как я могу сделать мир лучше? Этот вопрос направлял все мои решения — от выбора специальности в {university} до участия в волонтёрских проектах.

Работая над {project}, я столкнулся с проблемами, которые невозможно решить в одиночку. Это научило меня ценить командную работу и показало, как важно окружать себя единомышленниками.

Программа inVision U привлекает меня своим фокусом на инновации и предпринимательство. Я вижу, как выпускники создают стартапы и запускают социальные инициативы, и хочу стать частью этого сообщества.

В {city} я организовал{a} несколько мероприятий для молодёжи, что помогло мне развить навыки {skill} и укрепило мою решимость продолжать образование на высшем уровне.

Я стремлюсь к тому, чтобы {future_plan}. inVision U — это мост между моими мечтами и реальностью, и я готов{a} пройти по нему.""",
]

ESSAY_TEMPLATES_LEADERSHIP = [
    """Главный урок лидерства я получил{a}, когда возглавил{a} команду из 12 человек для организации {event} в нашем университете. На первый взгляд задача казалась простой, но реальность оказалась гораздо сложнее.

Первая проблема возникла уже на этапе планирования: участники команды имели разные представления о формате мероприятия. Вместо того чтобы навязать своё видение, я организовал{a} серию мозговых штурмов, где каждый мог высказаться.

Настоящее испытание пришло за неделю до события, когда наш основной спонсор отказался от участия. Мне пришлось быстро перестроить бюджет и найти альтернативные источники финансирования. Я обзвонил{a} более 20 компаний за два дня и нашёл{ла} троих новых партнёров.

Этот опыт научил меня нескольким важным вещам. Во-первых, настоящий лидер слушает свою команду. Во-вторых, гибкость и готовность к изменениям — это не слабость, а сила. В-третьих, кризис — это возможность проявить свои лучшие качества.

Сегодня я применяю эти уроки в каждом проекте. Руководя командой по разработке {project}, я стараюсь создавать атмосферу доверия и открытости, где каждый чувствует свою значимость.""",

    """Лидерство для меня — это не про должности и титулы. Это про ответственность и готовность действовать, когда другие сомневаются. Я осознал{а} это, когда работал{а} координатором волонтёрской программы в {city}.

Наша задача была амбициозной: организовать серию образовательных мастер-классов для 200 школьников из отдалённых районов. Бюджет был минимальным, а сроки — жёсткими.

Я начал{а} с формирования команды. Важно было найти людей, разделяющих нашу миссию, а не просто исполнителей. Мы разработали систему наставничества, где опытные волонтёры обучали новичков.

Самым сложным моментом стала логистика: как доставить материалы и обеспечить транспорт для волонтёров. Я решил{а} проблему, договорившись с местными предпринимателями о партнёрстве. Это потребовало навыков переговоров, которые я развивал{а} в дебатном клубе.

В результате мы провели 15 мастер-классов, охватив более 250 школьников. Но самым ценным результатом стала команда единомышленников, которая продолжает работать вместе и после завершения программы. Этот опыт укрепил моё понимание: лидер создаёт не последователей, а новых лидеров.""",
]

ESSAY_TEMPLATES_PROBLEM = [
    """Проблема, которая не даёт мне покоя — это неравный доступ к качественному образованию в Казахстане. Живя в {city}, я вижу контраст между возможностями в крупных городах и сельской местности.

По данным исследований, более 40% школьников в сельских районах не имеют доступа к высокоскоростному интернету. Это означает, что тысячи талантливых молодых людей лишены возможностей, которые их городские сверстники воспринимают как должное.

Я уже начал{а} работать над решением этой проблемы. Вместе с командой мы создали проект {project}, который позволяет предоставлять образовательный контент в оффлайн-режиме. Мы разработали систему, которая синхронизирует материалы при наличии даже слабого интернет-соединения.

Однако технологии — это только часть решения. Необходимо также обучать учителей работе с цифровыми инструментами и менять подход к образованию в целом. Именно поэтому я хочу углубить свои знания в inVision U.

Моя цель — создать масштабируемую EdTech-платформу, которая сделает качественное образование доступным для каждого ребёнка в Казахстане, независимо от его географического положения. Я верю, что технологии могут стать великим уравнителем.""",

    """Экологическая ситуация в Казахстане — одна из проблем, которая требует немедленного внимания. Наша страна сталкивается с серьёзными вызовами: от загрязнения воздуха в крупных городах до проблемы Аральского моря.

В {city} я лично наблюдаю последствия экологического кризиса. Загрязнение воздуха в зимние месяцы достигает критических отметок, что влияет на здоровье миллионов людей. При этом существующие решения зачастую неэффективны.

Мой подход к этой проблеме основан на данных. Работая над проектом {project}, я использовал{а} методы {field} для анализа экологических данных и выявления паттернов загрязнения. Результаты показали, что целенаправленные интервенции в ключевых точках могут значительно улучшить ситуацию.

Я убеждён{а}, что решение экологических проблем требует междисциплинарного подхода. Нужны специалисты, которые понимают и технологии, и экономику, и социальные процессы. Именно такое образование предлагает inVision U.

Мой план — создать систему мониторинга и прогнозирования экологической обстановки с использованием IoT-датчиков и машинного обучения. Это позволит принимать решения на основе данных, а не интуиции.""",
]

FUTURE_PLANS = [
    "создать технологический стартап, решающий проблемы образования в Центральной Азии",
    "разработать платформу для поддержки молодых предпринимателей в Казахстане",
    "стать ведущим специалистом в области искусственного интеллекта и применять знания для развития региона",
    "основать социальное предприятие, помогающее людям с ограниченными возможностями",
    "создать инновационный хаб, объединяющий технологии и креативные индустрии",
    "запустить программу менторства для студентов из малых городов",
    "разработать решения в сфере GreenTech для устойчивого развития Казахстана",
    "построить карьеру в международной технологической компании и вернуть экспертизу в Казахстан",
]

FIELDS = [
    "информационных технологий", "аналитики данных", "машинного обучения",
    "продуктового менеджмента", "цифрового маркетинга", "финтеха",
    "разработки ПО", "кибербезопасности", "UX/UI дизайна",
]

EVENTS = [
    "хакатон по социальным инновациям", "конференцию по EdTech",
    "фестиваль науки и технологий", "форум молодых предпринимателей",
    "благотворительный марафон", "студенческий TEDx",
]

SUBJECTS = ["математике", "физике", "информатике", "экономике", "биологии"]
SPORTS = ["шахматам", "лёгкой атлетике", "плаванию", "волейболу", "дзюдо"]
TOPICS = ["ИИ в медицине", "блокчейн-технологии", "урбанистика", "EdTech", "экология"]
CLUB_NAMES = ["инноваторов", "программирования", "социальных инициатив", "робототехники"]
HACKATHON_NAMES = ["HackNU", "Astana Hub Challenge", "Digital Bridge Hack", "TechOrd"]
APP_NAMES = ["EcoTrack KZ", "StudyBuddy", "QalaStar", "MedHelper"]
STARTUP_NAMES = ["EduBridge", "GreenStep KZ", "TechNomad", "DataPulse"]


def _fill_template(template: str, gender: str, city: str, university: str) -> str:
    a_suffix = "а" if gender == "f" else ""
    la_suffix = "ла" if gender == "f" else ""
    ла_suffix = "ла" if gender == "f" else ""
    а_suffix = "а" if gender == "f" else ""

    text = template.replace("{a}", a_suffix)
    text = text.replace("{ла}", ла_suffix)
    text = text.replace("{city}", city)
    text = text.replace("{university}", university)
    text = text.replace("{project}", random.choice(PROJECTS_POOL))
    text = text.replace("{field}", random.choice(FIELDS))
    text = text.replace("{future_plan}", random.choice(FUTURE_PLANS))
    text = text.replace("{event}", random.choice(EVENTS))
    text = text.replace("{skill}", random.choice(SKILLS_POOL))
    text = text.replace("{subject}", random.choice(SUBJECTS))
    text = text.replace("{sport}", random.choice(SPORTS))
    text = text.replace("{topic}", random.choice(TOPICS))
    text = text.replace("{club_name}", random.choice(CLUB_NAMES))
    text = text.replace("{hackathon_name}", random.choice(HACKATHON_NAMES))
    text = text.replace("{app_name}", random.choice(APP_NAMES))
    text = text.replace("{startup_name}", random.choice(STARTUP_NAMES))

    return text


def _generate_achievement(category: str) -> dict:
    templates = ACHIEVEMENT_TEMPLATES.get(category, ACHIEVEMENT_TEMPLATES["academic"])
    title = random.choice(templates)

    title = title.replace("{field}", random.choice(FIELDS))
    title = title.replace("{subject}", random.choice(SUBJECTS))
    title = title.replace("{topic}", random.choice(TOPICS))
    title = title.replace("{sport}", random.choice(SPORTS))
    title = title.replace("{club_name}", random.choice(CLUB_NAMES))
    title = title.replace("{hackathon_name}", random.choice(HACKATHON_NAMES))
    title = title.replace("{app_name}", random.choice(APP_NAMES))
    title = title.replace("{startup_name}", random.choice(STARTUP_NAMES))

    return {
        "title": title,
        "category": category,
        "year": random.randint(2020, 2025),
    }


def generate_candidate(candidate_id: int) -> dict:
    gender = random.choice(["m", "f"])

    if gender == "m":
        first_name = random.choice(FIRST_NAMES_M)
        last_name = random.choice(LAST_NAMES)
    else:
        first_name = random.choice(FIRST_NAMES_F)
        last_name = random.choice(LAST_NAMES) + "а"

    full_name = f"{last_name} {first_name}"
    email_name = f"{first_name.lower()}.{last_name.lower().rstrip('а')}"
    email = f"{email_name}@{'gmail.com' if random.random() > 0.3 else 'mail.ru'}"

    city = random.choice(CITIES)
    university = random.choice(UNIVERSITIES)
    age = random.randint(18, 28)

    edu_weights = {"school": 0.1, "bachelor": 0.5, "master": 0.3, "phd": 0.1}
    education_level = random.choices(
        list(edu_weights.keys()),
        weights=list(edu_weights.values()),
    )[0]

    quality = random.random()

    if quality > 0.8:
        gpa = round(random.uniform(3.5, 4.0), 2)
        num_skills = random.randint(5, 10)
        num_achievements = random.randint(3, 6)
        volunteer_hours = random.randint(100, 400)
        work_experience = round(random.uniform(1, 4), 1)
        num_projects = random.randint(2, 5)
        rec_count = random.randint(2, 4)
        num_languages = random.randint(2, 4)
        num_leadership = random.randint(1, 4)
    elif quality > 0.4:
        gpa = round(random.uniform(2.8, 3.6), 2)
        num_skills = random.randint(3, 7)
        num_achievements = random.randint(1, 4)
        volunteer_hours = random.randint(20, 150)
        work_experience = round(random.uniform(0, 2.5), 1)
        num_projects = random.randint(1, 3)
        rec_count = random.randint(1, 3)
        num_languages = random.randint(1, 3)
        num_leadership = random.randint(0, 2)
    else:
        gpa = round(random.uniform(2.0, 3.0), 2)
        num_skills = random.randint(1, 4)
        num_achievements = random.randint(0, 2)
        volunteer_hours = random.randint(0, 50)
        work_experience = round(random.uniform(0, 1), 1)
        num_projects = random.randint(0, 2)
        rec_count = random.randint(0, 1)
        num_languages = random.randint(1, 2)
        num_leadership = random.randint(0, 1)

    skills = random.sample(SKILLS_POOL, min(num_skills, len(SKILLS_POOL)))
    languages = random.sample(LANGUAGES_POOL, min(num_languages, len(LANGUAGES_POOL)))
    if "Казахский" not in languages:
        languages[0] = "Казахский"
    if "Русский" not in languages and len(languages) > 1:
        languages[1] = "Русский"

    achievement_cats = random.sample(
        ACHIEVEMENT_CATEGORIES,
        min(num_achievements, len(ACHIEVEMENT_CATEGORIES)),
    )
    achievements = [_generate_achievement(cat) for cat in achievement_cats]

    leadership_roles = random.sample(
        LEADERSHIP_ROLES_POOL,
        min(num_leadership, len(LEADERSHIP_ROLES_POOL)),
    )
    projects = random.sample(PROJECTS_POOL, min(num_projects, len(PROJECTS_POOL)))

    essays = []
    motivation_template = random.choice(ESSAY_TEMPLATES_MOTIVATION)
    essays.append({
        "prompt": ESSAY_PROMPTS[0],
        "text": _fill_template(motivation_template, gender, city, university),
        "language": "ru",
    })

    leadership_template = random.choice(ESSAY_TEMPLATES_LEADERSHIP)
    essays.append({
        "prompt": ESSAY_PROMPTS[1],
        "text": _fill_template(leadership_template, gender, city, university),
        "language": "ru",
    })

    problem_template = random.choice(ESSAY_TEMPLATES_PROBLEM)
    essays.append({
        "prompt": ESSAY_PROMPTS[2],
        "text": _fill_template(problem_template, gender, city, university),
        "language": "ru",
    })

    cid = hashlib.md5(f"candidate_{candidate_id}_{full_name}".encode()).hexdigest()[:12]

    return {
        "id": cid,
        "full_name": full_name,
        "email": email,
        "age": age,
        "city": city,
        "country": "Казахстан",
        "education_level": education_level,
        "university": university,
        "gpa": gpa,
        "work_experience_years": work_experience,
        "skills": skills,
        "achievements": achievements,
        "essays": essays,
        "languages": languages,
        "leadership_roles": leadership_roles,
        "volunteer_hours": volunteer_hours,
        "projects": projects,
        "recommendation_count": rec_count,
    }


def generate_dataset(num_candidates: int = 55, seed: int = 42) -> list[dict]:
    random.seed(seed)
    candidates = [generate_candidate(i) for i in range(num_candidates)]
    return candidates


def save_dataset(candidates: list[dict], path: str = "sample_data/candidates.json"):
    output = {"candidates": candidates, "metadata": {
        "generated_count": len(candidates),
        "version": "1.0",
        "description": "Синтетические данные кандидатов для inVision U",
    }}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    candidates = generate_dataset(55)
    path = save_dataset(candidates)
    print(f"Сгенерировано {len(candidates)} кандидатов -> {path}")
