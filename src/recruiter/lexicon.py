"""Skill / role / domain lexicons shared by C1, C2, C3, C6.

Canonical skill -> aliases (lower-case surface forms). Matching is whole-token via regex,
so "go" does not match "google" and "c" is excluded (too ambiguous).
"""

from __future__ import annotations

import re
from functools import lru_cache

SKILLS: dict[str, list[str]] = {
    # languages
    "Python": ["python", "python3"],
    "Java": ["java", "java 8", "java 11", "java 17"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "C#": ["c#", "csharp", "c sharp"],
    "C++": ["c++", "cpp"],
    "Go": ["golang", "go lang"],
    "Rust": ["rust"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "Scala": ["scala"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift", "swiftui"],
    "Objective-C": ["objective-c", "objective c"],
    "Dart": ["dart"],
    "R": ["r language", "rstudio"],
    "SQL": ["sql", "t-sql", "pl/sql", "tsql"],
    "Bash": ["bash", "shell scripting", "shell"],
    "Solidity": ["solidity"],
    "MATLAB": ["matlab"],
    # web / frameworks
    "React": ["react", "reactjs", "react.js"],
    "React Native": ["react native", "react-native"],
    "Redux": ["redux"],
    "Angular": ["angular", "angularjs"],
    "Vue.js": ["vue", "vuejs", "vue.js", "nuxt", "nuxt.js"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node", "nodejs", "node.js"],
    "Express": ["express", "express.js", "expressjs"],
    "NestJS": ["nestjs", "nest.js"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3", "scss", "sass", "less"],
    "Tailwind": ["tailwind", "tailwindcss"],
    "jQuery": ["jquery"],
    "Webpack": ["webpack"],
    "GraphQL": ["graphql"],
    "REST API": ["rest", "rest api", "restful", "restful api"],
    "gRPC": ["grpc"],
    "Django": ["django", "drf", "django rest framework"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring": ["spring", "spring boot", "springboot", "spring framework"],
    "Hibernate": ["hibernate", "jpa"],
    ".NET": [".net", "dotnet", ".net core", "asp.net", "asp.net core"],
    "Entity Framework": ["entity framework", "ef core"],
    "Laravel": ["laravel"],
    "Symfony": ["symfony"],
    "WordPress": ["wordpress"],
    "Ruby on Rails": ["rails", "ruby on rails", "ror"],
    "Flutter": ["flutter"],
    "Unity": ["unity", "unity3d"],
    "Unreal Engine": ["unreal", "unreal engine"],
    "Android": ["android", "android sdk"],
    "iOS": ["ios"],
    "Jetpack Compose": ["jetpack compose"],
    # data / ml
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "opencv", "cv engineer", "cv engineering"],
    "LLM": ["llm", "llms", "large language models", "gpt", "langchain"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop", "hdfs"],
    "Airflow": ["airflow", "apache airflow"],
    "Kafka": ["kafka", "apache kafka"],
    "dbt": ["dbt"],
    "ETL": ["etl", "elt"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "ms excel"],
    "Statistics": ["statistics", "statistical analysis"],
    "Data Analysis": ["data analysis", "data analytics"],
    "A/B Testing": ["a/b testing", "ab testing", "a/b tests"],
    # databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql", "mariadb"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search", "elk"],
    "MS SQL": ["ms sql", "mssql", "sql server"],
    "Oracle": ["oracle", "oracle db"],
    "DynamoDB": ["dynamodb"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery"],
    "ClickHouse": ["clickhouse"],
    # devops / cloud
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "GCP": ["gcp", "google cloud"],
    "Azure": ["azure"],
    "Docker": ["docker", "docker-compose", "docker compose"],
    "Kubernetes": ["kubernetes", "k8s", "helm"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "CI/CD": ["ci/cd", "ci cd", "cicd", "continuous integration"],
    "Jenkins": ["jenkins"],
    "GitLab CI": ["gitlab ci", "gitlab"],
    "GitHub Actions": ["github actions"],
    "Linux": ["linux", "ubuntu", "centos", "debian"],
    "Nginx": ["nginx"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "Git": ["git"],
    "Microservices": ["microservices", "microservice", "microservice architecture"],
    "RabbitMQ": ["rabbitmq"],
    # qa
    "Manual Testing": ["manual testing", "manual qa", "test cases", "test case"],
    "Test Automation": ["test automation", "automation testing", "automated testing"],
    "Selenium": ["selenium", "selenium webdriver"],
    "Cypress": ["cypress"],
    "Playwright": ["playwright"],
    "Postman": ["postman"],
    "JMeter": ["jmeter"],
    "Appium": ["appium"],
    "API Testing": ["api testing"],
    "Jira": ["jira"],
    "TestRail": ["testrail"],
    # design / product / mgmt
    "Figma": ["figma"],
    "Sketch": ["sketch"],
    "Adobe Photoshop": ["photoshop"],
    "Adobe Illustrator": ["illustrator"],
    "UI/UX": ["ui/ux", "ux/ui", "ui", "ux", "user experience", "user interface"],
    "Prototyping": ["prototyping", "wireframes", "wireframing"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Kanban": ["kanban"],
    "Project Management": ["project management"],
    "Product Management": ["product management", "product discovery"],
    "Stakeholder Management": ["stakeholder management", "stakeholders"],
    "Requirements Analysis": ["requirements analysis", "requirements gathering", "bpmn", "uml"],
    "Team Leadership": ["team lead", "team leadership", "people management", "mentoring"],
    # marketing / sales / hr
    "SEO": ["seo", "search engine optimization"],
    "PPC": ["ppc", "google ads", "adwords"],
    "SMM": ["smm", "social media marketing"],
    "Google Analytics": ["google analytics", "ga4"],
    "Content Marketing": ["content marketing", "copywriting"],
    "Lead Generation": ["lead generation", "lead gen"],
    "CRM": ["crm", "hubspot", "pipedrive"],
    "Salesforce": ["salesforce"],
    "B2B Sales": ["b2b sales", "b2b"],
    "Recruiting": ["recruiting", "recruitment", "sourcing", "headhunting"],
    "Technical Writing": ["technical writing", "documentation"],
    "Cybersecurity": ["cybersecurity", "penetration testing", "pentest", "owasp", "siem"],
    "Blockchain": ["blockchain", "web3", "smart contracts", "ethereum"],
    "SAP": ["sap", "abap"],
}

# Djinni Primary Keyword -> phrases recruiters use for the role
ROLES: dict[str, list[str]] = {
    "Python": ["python developer", "python engineer", "python"],
    "Java": ["java developer", "java engineer", "java"],
    "JavaScript": [
        "javascript developer",
        "front-end",
        "frontend",
        "front end",
        "javascript",
        "react developer",
        "angular developer",
        "vue developer",
    ],
    "Node.js": ["node.js developer", "node developer", "nodejs developer", "node.js", "backend javascript"],
    ".NET": [".net developer", "c# developer", ".net", "c#"],
    "PHP": ["php developer", "php", "laravel developer"],
    "Golang": ["go developer", "golang developer", "golang"],
    "Ruby": ["ruby developer", "rails developer", "ruby"],
    "Rust": ["rust developer", "rust"],
    "Scala": ["scala developer", "scala"],
    "C++": ["c++ developer", "c++"],
    "iOS": ["ios developer", "ios engineer", "swift developer", "ios"],
    "Android": ["android developer", "kotlin developer", "android"],
    "Flutter": ["flutter developer", "flutter"],
    "Unity": ["unity developer", "game developer", "unity"],
    "QA": ["manual qa", "qa engineer", "tester", "quality assurance", "qa"],
    "QA Automation": ["qa automation", "automation qa", "aqa", "sdet", "test automation engineer"],
    "DevOps": ["devops", "sre", "site reliability", "platform engineer", "cloud engineer"],
    "Sysadmin": ["system administrator", "sysadmin", "network administrator"],
    "Data Science": [
        "data scientist",
        "ml engineer",
        "machine learning engineer",
        "data science",
        "ai engineer",
        "nlp engineer",
        "cv engineer",
        "computer vision engineer",
        "deep learning engineer",
        "research engineer",
        "ml researcher",
    ],
    "Data Engineer": ["data engineer", "big data engineer", "etl developer"],
    "Data Analyst": ["data analyst", "bi analyst", "analytics engineer"],
    "SQL": ["sql developer", "database developer", "dba"],
    "Business Analyst": ["business analyst", "system analyst"],
    "Product Manager": ["product manager"],
    "Product Owner": ["product owner"],
    "Project Manager": ["project manager", "delivery manager"],
    "Scrum Master": ["scrum master", "agile coach"],
    "Design": ["designer", "ui/ux", "ux designer", "ui designer", "product designer", "web designer"],
    "Artist": ["artist", "2d artist", "3d artist", "illustrator", "animator"],
    "Marketing": ["marketing manager", "marketing specialist", "digital marketing", "marketer", "smm manager", "ppc specialist"],
    "SEO": ["seo specialist", "seo"],
    "Lead Generation": ["lead generation", "lead generator"],
    "Sales": ["sales manager", "account manager", "account executive", "business development manager", "sales representative"],
    "Salesforce": ["salesforce developer", "salesforce"],
    "HR": ["hr manager", "hr generalist", "hr business partner", "people partner"],
    "Recruiter": ["recruiter", "talent acquisition", "sourcer"],
    "Support": ["support engineer", "customer support specialist", "technical support engineer", "help desk"],
    "Security": ["security engineer", "pentester", "security analyst", "cybersecurity engineer"],
    "Technical Writing": ["technical writer"],
    "Lead": ["tech lead", "team lead", "cto", "engineering manager", "head of engineering"],
    "Block-chain": ["blockchain developer", "solidity developer", "web3 developer"],
    "SAP": ["sap consultant", "abap developer"],
    "React": ["react developer"],
    "Other": [],
}

# Roles whose candidates are reasonable for a job of the key role (C3 role match)
RELATED_ROLES: dict[str, set[str]] = {
    "JavaScript": {"JavaScript", "React", "Node.js"},
    "React": {"React", "JavaScript"},
    "Node.js": {"Node.js", "JavaScript"},
    "QA Automation": {"QA Automation", "QA"},
    "QA": {"QA", "QA Automation"},
    "Data Science": {"Data Science", "Data Engineer", "Data Analyst"},
    "Data Engineer": {"Data Engineer", "Data Science", "SQL"},
    "Data Analyst": {"Data Analyst", "SQL", "Data Science", "Business Analyst"},
    "Product Manager": {"Product Manager", "Product Owner", "Project Manager"},
    "Product Owner": {"Product Owner", "Product Manager"},
    "Project Manager": {"Project Manager", "Scrum Master", "Product Manager"},
    "Scrum Master": {"Scrum Master", "Project Manager"},
    "DevOps": {"DevOps", "Sysadmin"},
    "Sysadmin": {"Sysadmin", "DevOps", "Support"},
    "Lead": {"Lead"},  # filled dynamically: any engineering role
    "Recruiter": {"Recruiter", "HR"},
    "HR": {"HR", "Recruiter"},
    "Marketing": {"Marketing", "SEO", "Lead Generation"},
    "Block-chain": {"Block-chain", "JavaScript", "Golang", "Rust"},
}

DOMAINS: dict[str, list[str]] = {
    "fintech": ["fintech", "banking", "payments", "payment", "trading", "finance"],
    "healthcare": ["healthcare", "medtech", "medical", "health"],
    "e-commerce": ["e-commerce", "ecommerce", "retail", "marketplace"],
    "gamedev": ["gamedev", "game development", "games", "gaming"],
    "gambling": ["gambling", "igaming", "casino", "betting"],
    "edtech": ["edtech", "education", "e-learning"],
    "adtech": ["adtech", "advertising"],
    "logistics": ["logistics", "supply chain"],
    "telecom": ["telecom", "telecommunications"],
    "blockchain": ["blockchain", "crypto", "web3", "defi"],
    "saas": ["saas", "b2b saas"],
    "cybersecurity": ["cybersecurity", "infosec"],
    "travel": ["travel", "hospitality"],
    "automotive": ["automotive"],
    "real estate": ["real estate", "proptech"],
    "r&d": ["r&d", "research and development", "research project"],
}

ENGLISH_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:fluent|native|proficient|c1|c2|advanced)\b", "fluent"),
    (r"\b(?:upper[- ]?intermediate|b2)\b", "upper"),
    (r"\b(?:pre[- ]?intermediate|a2)\b", "pre"),
    (r"\b(?:intermediate|b1)\b", "intermediate"),
    (r"\b(?:basic|elementary|beginner|a1)\b", "basic"),
]

SENIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:lead|principal|head of|architect|staff engineer|cto)\b", "lead"),
    (r"\b(?:senior|sr\.?)\b", "senior"),
    (r"\b(?:middle|mid[- ]level|mid)\b", "middle"),
    (r"\b(?:junior|jr\.?|trainee|intern|entry[- ]level)\b", "junior"),
]


def _alias_regex(alias: str) -> str:
    esc = re.escape(alias)
    # whole-token boundaries that work with symbols like c++, .net, c#
    return rf"(?<![\w+#.]){esc}(?![\w+#])"


@lru_cache(maxsize=1)
def _compiled_skills() -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for canon, aliases in SKILLS.items():
        pat = "|".join(_alias_regex(a) for a in sorted(set(aliases), key=len, reverse=True))
        out.append((canon, re.compile(pat, re.IGNORECASE)))
    return out


@lru_cache(maxsize=1)
def _alias_to_canon() -> dict[str, str]:
    m: dict[str, str] = {}
    for canon, aliases in SKILLS.items():
        m[canon.lower()] = canon
        for a in aliases:
            m.setdefault(a.lower(), canon)
    return m


# ambiguous short aliases only count when written in a technical casing
_CASE_SENSITIVE = {"go": "Go", "ts": "TS", "ui": "UI", "ux": "UX", "ml": "ML", "r": "R", "rest": "REST"}
# English words that are also tech names: only count when capitalised ("Spring", not "spring")
_CAPITALISED = {
    "express",
    "spring",
    "sketch",
    "unity",
    "spark",
    "excel",
    "swift",
    "rust",
    "lambda",
    "oracle",
    "ruby",
    "dart",
    "shell",
    "flask",
    "illustrator",
    "documentation",
    "torch",
}


def _accept(surface: str) -> bool:
    low = surface.lower()
    if low in _CASE_SENSITIVE:
        return surface == _CASE_SENSITIVE[low]
    if low in _CAPITALISED:
        return surface[:1].isupper()
    return True


def extract_skills(text: str) -> list[str]:
    """Return canonical skills found in text (order of first appearance)."""
    if not text:
        return []
    found: list[tuple[int, str]] = []
    spans = [(m.start(), m.end(), canon) for canon, pat in _compiled_skills() for m in pat.finditer(text) if _accept(m.group(0))]
    inside = {(a, b, c) for (a, b, c) in spans for (x, y, d) in spans if d != c and x <= a and b <= y and (y - x) > (b - a)}
    for canon, pat in _compiled_skills():
        for m in pat.finditer(text):
            surface = m.group(0)
            if not _accept(surface) or (m.start(), m.end(), canon) in inside:
                continue
            if canon == "React" and text[m.end() : m.end() + 7].lower().lstrip(" -").startswith("native"):
                continue
            found.append((m.start(), canon))
            break
    return [c for _, c in sorted(found)]


def find_skill_span(text: str, skill: str) -> tuple[int, int] | None:
    canon = normalize_skill(skill) or skill
    for c, pat in _compiled_skills():
        if c == canon:
            for m in pat.finditer(text):
                if _accept(m.group(0)):
                    return (m.start(), m.end())
            return None
    m = re.search(_alias_regex(skill.lower()), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


@lru_cache(maxsize=100_000)
def normalize_skill(name: str) -> str | None:
    """Map any surface form to a canonical skill (None if unknown)."""
    key = name.strip().lower()
    if key in _alias_to_canon():
        return _alias_to_canon()[key]
    hits = extract_skills(name)
    return hits[0] if len(hits) == 1 else None


def is_known_skill(name: str) -> bool:
    return normalize_skill(name) is not None


# Roles named after a programming language/framework. When a text also names a job type
# ("AQA Python", "Data Engineer (Python)"), the job type is the role and the language a skill.
LANGUAGE_ROLES = {"Python", "Java", "JavaScript", "Node.js", ".NET", "PHP", "Golang", "Ruby", "Rust", "Scala", "C++", "React"}


def detect_role(text: str) -> str | None:
    """Most specific Djinni role in text: job-type roles beat language roles, then longest phrase wins."""
    low = text.lower()
    best: tuple[int, int, str] | None = None
    for role, phrases in ROLES.items():
        for p in phrases:
            if re.search(rf"(?<![\w+#.]){re.escape(p)}(?![\w+#])", low):
                key = (int(role not in LANGUAGE_ROLES), len(p), role)
                if best is None or key[:2] > best[:2]:
                    best = key
    return best[2] if best else None


def detect_domains(text: str) -> list[str]:
    low = text.lower()
    return [d for d, kws in DOMAINS.items() if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws)]


def detect_english(text: str) -> str | None:
    low = text.lower()
    if "english" not in low and not re.search(r"\b(?:b1|b2|c1|c2|a2)\b", low):
        return None
    # look near the word "english" first
    windows = [low[max(0, m.start() - 40) : m.end() + 40] for m in re.finditer("english", low)] or [low]
    for w in windows:
        for pat, level in ENGLISH_PATTERNS:
            if re.search(pat, w):
                return level
    return None


def detect_seniority(text: str) -> str | None:
    low = text.lower()
    for pat, level in SENIORITY_PATTERNS:
        if re.search(pat, low):
            return level
    return None


def related_roles(role: str) -> set[str]:
    if role == "Lead":
        return set(ROLES) - {"HR", "Recruiter", "Sales", "Marketing", "Lead Generation", "SEO", "Artist"}
    return RELATED_ROLES.get(role, {role})
