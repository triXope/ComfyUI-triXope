import sys

RACE_LIST = [
    "Australoid", "Bugbear", "Centaur", "Dhampir", "Dragonborn", "Dwarf", "Elf", "Extraterrestrial", "Fairy", "Giant", "Gnome", "Goblin", "Goliath", "Half-Elf", "Half-Orc", "Halfling", "Hobgoblin", "Human - Mongoloid/Asian", "Human - Negroid/Black",
    "Human - White/Caucasian", "Kobold", "Lizardfolk", "Minotaur", "Orc", "Satyr", "Tiefling", "Vampire", "Werewolf", "Other/Unknown"
]

ROLE_LIST = [
    "Protagonist", "Protagonist's Helper", "Sidekick", "Guardian", "Mentor",
    "Catalyst", "Impact", "Antagonist", "Antagonist's Helper", "Skeptic",
    "Obstacle", "Goal", "Minor Character", "Other/Unknown"
]

EYECOLOR_LIST = [
    "Amber", "Blue", "Brown", "Gray", "Green", "Hazel", "Red", "Yellow",
    "Black", "Other/Unknown"
]

SKINTONE_LIST = [
    "Fair", "Light", "Medium", "Olive", "Tan", "Brown", "Dark Brown",
    "Black", "Other/Unknown"
]

BODYTYPE_LIST = [
    "Athletic", "Ectomorph", "Mesomorph", "Endomorph", "Rectangle", "Triangle", "Spoon",
    "Hourglass", "Top Hourglass", "Bottom Hourglass", "Inverted Triangle",
    "Round or Oval", "Diamond", "Other/Unknown"
]

POSTURE_LIST = [
    "Healthy", "Kyphosis", "Flat Back", "Swayback (Lordosis)",
    "Forward Neck or Head", "Other/Unknown"
]

FACESHAPE_LIST = [
    "Oval", "Round", "Square", "Diamond", "Heart", "Pear", "Oblong",
    "Other/Unknown"
]

HAIRSTYLE_LIST = [
    "Afro", "Asymmetric Cut", "Beehive", "Bald", "Bangs", "Big Hair", "Blowout",
    "Bob Cut", "Bouffant", "Bowl Cut", "Braid", "Brush Cut", "Bun", "Bunches",
    "Burr", "Businessman", "Butch Cut", "Buzz Cut", "Caesar Cut", "Chignon",
    "Chonmage", "Comb Over", "Comma Hair", "Conk", "Cornrows", "Crew Cut",
    "Crop", "Crown Braid", "Croydon Facelift", "Curtained Hair", "Devilock",
    "Dido Flip", "Double Buns", "Dreadlocks", "Duck's Ass", "Eton Crop",
    "Extensions", "Fade", "Fallera Hairdo", "Fauxhawk", "Feather Cut",
    "Feathered Hair", "Finger Waves", "Fishtail Hair", "Flattop",
    "Flipped-up Ends", "Fontange", "French Braid", "French Twist",
    "Fringe (Bangs)", "Frosted Tips", "Full Crown", "Half Crown", "Half Updo",
    "Harvard Clip", "High and Tight", "Highlights", "Hime Cut",
    "Historic Hairstyle", "Hi-Top Fade", "Induction Cut", "Ivy League",
    "Jewfro", "Jheri Curl", "Layered Hair", "Liberty Spikes", "Line Up", "Lob",
    "Long Hair", "Marcel Waves", "Mod Cut", "Mohawk", "Mop-Top", "Mullet",
    "Natural", "Odango", "Oseledets", "Pageboy", "Payot", "Perm", "Pigtails",
    "Pixie Cut", "Pompadour", "Ponyhawk", "Ponytail", "Princeton",
    "Professional Cut", "Psychobilly Wedge", "Queue", "Quiff", "The Rachel",
    "Rattail", "Razor Cut", "Regular Haircut", "Regular Taper Cut", "Ringlets",
    "Shag Cut", "Shape-Up", "Shingle Bob", "Short Back and Sides",
    "Short Brush Cut", "Short Hair", "Slicked-Back", "Spiky Hair",
    "Standard Haircut", "Step Cut", "Surfer Hair", "Taper Cut", "Tail on Back",
    "Tonsure", "Undercut", "Updo", "Waves", "Weave", "Wings", "Other/Unknown"
]

HAIRCOLOR_LIST = [
    "Auburn", "Black", "Blonde", "Blue", "Brown", "Brunette", "Copper", "Gray",
    "Green", "None/Bald", "Pink", "Purple", "Red", "Silver", "White", "Other/Unknown"
]

WARDROBE_LIST = [
    "Streetwear", "Ethnic Fashion", "Formal Office Wear", "Business Casual",
    "Corporate Powerhouse", "Evening Black Tie", "Sports Wear", "Girly",
    "Rockstar", "Preppy", "Skateboarder", "Jock", "Maternity", "Goth",
    "Lolita", "Gothic Lolita", "Hip-Hop", "Chave Culture", "Kawaii",
    "Cowgirl/boy", "Lagenlook", "Fashion", "Scene Fashion", "Girl Next Door",
    "Casual", "Geeky", "Military", "Retro", "Flapper (20’s look)", "Tomboy",
    "Garconne Look", "Vacation (Resort)", "Artsy Fashion", "Grunge", "Punk",
    "Boho/Bohemian", "Psychedelic", "Cosplay", "Haute Couture", "Modest",
    "Rave", "Flamboyant", "Adventurer", "Ankara", "Other/Unknown"
]

LIFESKILL_LIST = [
    "Active Listening", "Adaptability", "Blending In", "Breath Control",
    "Charm", "Clairvoyance", "Communication", "Creativity", "Customer +",
    "Decisiveness", "Empathy", "Senses +", "Foraging", "Haggling",
    "Hospitality", "Humorous", "Leadership", "Lip-Reading", "Logical",
    "Lying", "Management", "Memorization", "Mimicking", "Multitasking",
    "Musicality", "Navigation", "Organization", "Problem-Solving", "Psychic",
    "Psychotic", "Public Speaking", "Regeneration", "Self-Defense",
    "Sharpshooting", "Sleight-of-Hand", "Strength", "Survival Skills",
    "Swifted-Footed", "Teamwork", "Telekinesis", "Throws Voice", "Trustworthy", "Other/Unknown"
]

LIFESTRENGTH_LIST = [
    "Brave", "Reckless", "Idealistic", "Naive", "Funny", "Unserious",
    "Confident", "Arrogant", "Careful", "Timid", "Virtuous", "Rigid",
    "Open-Minded", "Overthinking", "Determined", "Stubborn", "Loyal",
    "Insular", "Humble", "Passive", "Patient", "Unproductive", "Energetic",
    "Exhausting", "Independent", "Poor Teamwork", "Imaginative",
    "Unrealistic", "Charismatic", "Self-Absorbed", "Emotional", "Illogical",
    "Proud", "Vain", "Driven", "Greedy", "Logical", "Low Empathy", "Honest",
    "Blunt", "Decisive", "Impulsive", "Leadership", "Bossy", "Focused",
    "Addictive", "Curious", "Aimless", "Kind", "Unassertive", "Connoisseur",
    "Glutton", "Content", "Unambitious", "Principled", "Intolerant",
    "Charming", "Deceitful", "Adaptable", "Unstructured", "Realistic",
    "Cynical", "Other/Unknown"
]

HOBBY_LIST = [
    "None", "Animal Watch", "Art", "Astronomy", "Billiards", "Bowling", "Camping",
    "Collecting", "Cooking", "Crafts", "Dance", "Darts", "Explosives",
    "Fishing", "Gaming", "Gambling", "Gardening", "Hunting", "Magic", "Movies",
    "Music", "Parkour", "Photography", "Puzzles", "Racing", "Reading",
    "Sewing", "Skating", "Sports/Fitness", "Travelling", "Writing",
    "Other/Unknown"
]

MUSICALINTEREST_LIST = [
    "None", "Alternative", "Blues", "Children's Music", "Classical", "Comedy",
    "Country", "Dance", "Disco", "Easy Listening", "Electronic", "Enka",
    "Folk", "Hip-Hop/Rap", "Holiday", "Indie", "Industrial", "Inspirational",
    "Instrumental", "Jazz", "Latin", "Opera", "Pop", "Progressive",
    "R&B/Soul", "Reggae", "Rock", "World", "Other/Unknown"
]

BADHABIT_LIST = [
    "None", "Abusive", "Alcoholic", "Avoids Eyes", "Belches", "Breaks Promises",
    "Cheats", "Chews Tobacco", "Chews Loudly", "Cracks Joints", "Daydreams",
    "Farts", "Fidgets", "Freeloads", "Gambles", "Gossips", "Grinds Teeth",
    "Hoarder", "Hums", "Indecisive", "Leaves Lid Up", "Licks Lips", "Litters",
    "Lies", "Mimicking", "Multitasks", "Nail Bites", "Negative",
    "Night Snacks", "Nose Picks", "Obsesses", "Overeats", "Over-medicates",
    "Oversleeps", "Overspends", "Picks Scabs", "Picks Teeth", "Plays w/ Hair",
    "Pops Gum", "Pops Zits", "Procrastinates", "Public Affection", "Sedentary",
    "Slang", "Slouches", "Smokes", "Speeds", "Spits", "Steals", "Stereotypes",
    "Swears", "Taps Foot", "Tardy", "Views Porn", "Violent", "Other/Unknown"
]

FAITH_LIST = [
    "Non-Spiritual", "Atheism", "Baha'i", "Buddhism", "Candomble", "Christianity", "Hinduism",
    "Islam", "Jainism", "Jehovah's Witness", "Judaism", "Mormonism",
    "Paganism", "Rastafari", "Santeria", "Shinto", "Sikhism", "Spiritualism",
    "Taoism", "Unitarianism", "Zoroastrianism", "Other/Unknown"
]

PET_LIST = [
    "None", "Alpaca", "Ant Farm", "Bird", "Cat", "Dog", "Ferret", "Fish",
    "Frog/Toad", "Gecko", "Gerbil", "Goat", "Guinea Pig", "Hamster", "Hedgehog",
    "Hermit Crab", "Horse", "Iguana", "Mantis", "Mouse", "Newt", "Pig",
    "Rabbit", "Rat", "Salamander", "Sheep", "Skunk", "Snake", "Spider",
    "Stick-Bug", "Turtle", "Other/Unknown"
]

PERSONALITYTYPE_LIST = [
    "The Architect/Thinker", "The Champion", "The Commander", "The Composer",
    "The Counselor", "The Craftsman", "The Dynamo/Doer", "The Healer/Idealist",
    "The Inspector", "The Mastermind", "The Performer",
    "The Protector/Nurturer", "The Provider", "The Supervisor",
    "The Teacher/Giver", "The Visionary", "Other/Unknown"
]

PHOBIA_LIST = [
    "None", "Achluophobia - darkness", "Acrophobia - heights",
    "Aerophobia - flying", "Algophobia - pain", "Alektorophobia - chickens",
    "Agoraphobia - crowds", "Aichmophobia - needles",
    "Amaxophobia - riding in a car", "Androphobia - men",
    "Anginophobia - choking", "Anthophobia - flowers",
    "Anthropophobia - people", "Aphenphosmphobia - being touched",
    "Arachnophobia - spiders", "Arithmophobia - numbers",
    "Astraphobia - thunder & lightning", "Ataxophobia - disorder",
    "Atelophobia - imperfection", "Atychiphobia - failure",
    "Autophobia - being alone", "Bacteriophobia - bacteria",
    "Barophobia - gravity", "Bathmophobia - steep slopes",
    "Batrachophobia - amphibians", "Belonephobia - pins & needles",
    "Bibliophobia - books", "Botanophobia - plants", "Cacophobia - ugliness",
    "Catagelophobia - being ridiculed", "Catoptrophobia - mirrors",
    "Chionophobia - snow", "Chromophobia - colors",
    "Chronomentrophobia - clocks", "Claustrophobia - confined spaces",
    "Coulrophobia - clowns", "Cyberphobia - computers", "Cynophobia - dogs",
    "Dendrophobia - trees", "Dentophobia - dentists", "Domatophobia - houses",
    "Dystychiphobia - accidents", "Ecophobia - home", "Elurophobia - cats",
    "Entomophobia - insects", "Ephebiphobia - teenagers",
    "Equinophobia - horses", "Gamophobia - marriage", "Genuphobia - knees",
    "Glossophobia - public speaking", "Gynophobia - women", "Heliophobia - sun",
    "Hemophobia - blood", "Herpetophobia - reptiles", "Hydrophobia - water",
    "Hypochondria - illness", "Iatrophobia - doctors",
    "Insectophobia - insects", "Koinoniphobia - rooms full of people",
    "Leukophobia - color white", "Lilapsophobia - tornadoes & hurricanes",
    "Lockiophobia - childbirth", "Mageirocophobia - cooking",
    "Megalophobia - large things", "Melanophobia - color black",
    "Microphobia - small things", "Mysophobia - dirt & germs",
    "Necrophobia - death or dead things", "Noctiphobia - night",
    "Nosocomephobia - hospitals", "Nyctophobia - the dark",
    "Obesophobia - gaining weight", "Octophobia - figure 8",
    "Ombrophobia - rain", "Ophidiophobia - snakes", "Ornithophobia - birds",
    "Papyrophobia - paper", "Pathophobia - disease", "Pedophobia - children",
    "Philophobia - love", "Phobophobia - phobias", "Podophobia - feet",
    "Pogonophobia - beards", "Porphyrophobia - color purple",
    "Pteridophobia - ferns", "Pteromerhanophobia - flying", "Pyrophobia - fire",
    "Samhainophobia - Halloween", "Scolionophobia - school",
    "Selenophobia - the moon", "Sociophobia - social evaluation",
    "Somniphobia - sleep", "Tachophobia - speed", "Technophobia - technology",
    "Tonitrophobia - thunder", "Trypanophobia - needles or injections",
    "Venustraphobia - beautiful women", "Verminophobia - germs",
    "Wiccaphobia - witches & witchcraft", "Xenophobia - strangers or foreigners",
    "Zoophobia - animals", "Other/Unknown"
]

OCCUPATION_LIST = [
    "Unemployed", "Computers & Technology", "Health Care", "Education & Social Services",
    "Arts & Communication", "Trades & Transportation",
    "Management, Business, & Finance", "Architecture & Engineering",
    "Farming & Agriculture", "Science", "Hospitality & Tourism",
    "Law & Enforcement", "Other/Unknown"
]

INCOME_LIST = [
    "$0 - $10,000", "$10,000 - $20,000", "$20,000 - $30,000",
    "$30,000 - $40,000", "$40,000 - $50,000", "$50,000 - $60,000",
    "$60,000 - $70,000", "$70,000 - $80,000", "$80,000 - $90,000",
    "$90,000 - $100,000", "$100,000 - $200,000", "$200,000 - $300,000",
    "$300,000 - $400,000", "$400,000 - $500,000", "$500,000 - $600,000",
    "$600,000 - $700,000", "$700,000 - $800,000", "$800,000 - $900,000",
    "$900,000 - $1,000,000", "greater than $1,000,000", "Other/Unknown"
]

SEXUALORIENTATION_LIST = [
    "Straight", "Gay", "Bi-Sexual", "Other/Unknown"
]

MARITALSTATUS_LIST = [
    "Married", "Single", "Divorced", "Separated", "Widowed", "Other/Unknown"
]

RELATIONTOPROTAGONIST_LIST = [
    "Self", "Grandparent", "Parent", "Sibling", "Child", "Cousin",
    "Aunt/Uncle", "Niece/Nephew", "Significant Other", "Other Relative",
    "Friend", "Nemesis", "Other/Unknown"
]

FIRSTAPPEARANCE_LIST = [
    "Act I (Beginning)", "Act II (Middle)", "Act III (End)", "Exposition",
    "Inciting Incident", "Rising Action", "Climax", "Falling Action",
    "Moment of Final Suspense", "Resolution", "Other/Unknown"
]


class TriXopeCharacterCraft:
    """
    A custom node for ComfyUI to craft character details.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "Name": ("STRING", {"multiline": False, "default": ""}),
                "Age": ("INT", {"default": 24, "min": 0, "max": 10000, "step": 1}),
                "Race": (RACE_LIST, {"default": RACE_LIST[0]}),
                "Role": (ROLE_LIST, {"default": ROLE_LIST[0]}),
                "Eye Color": (EYECOLOR_LIST, {"default": EYECOLOR_LIST[0]}),
                "Skin Tone": (SKINTONE_LIST, {"default": SKINTONE_LIST[0]}),
                "Body Type": (BODYTYPE_LIST, {"default": BODYTYPE_LIST[0]}),
                "Posture": (POSTURE_LIST, {"default": POSTURE_LIST[0]}),
                "Face Shape": (FACESHAPE_LIST, {"default": FACESHAPE_LIST[0]}),
                "Hair Style": (HAIRSTYLE_LIST, {"default": HAIRSTYLE_LIST[0]}),
                "Hair Color": (HAIRCOLOR_LIST, {"default": HAIRCOLOR_LIST[0]}),
                "Wardrobe": (WARDROBE_LIST, {"default": WARDROBE_LIST[0]}),
                "Life Skill": (LIFESKILL_LIST, {"default": LIFESKILL_LIST[0]}),
                "Life Strength": (LIFESTRENGTH_LIST, {"default": LIFESTRENGTH_LIST[0]}),
                "Hobby": (HOBBY_LIST, {"default": HOBBY_LIST[0]}),
                "Musical Interest": (MUSICALINTEREST_LIST, {"default": MUSICALINTEREST_LIST[0]}),
                "Bad Habit": (BADHABIT_LIST, {"default": BADHABIT_LIST[0]}),
                "Faith": (FAITH_LIST, {"default": FAITH_LIST[0]}),
                "Pet": (PET_LIST, {"default": PET_LIST[0]}),
                "Personality Type": (PERSONALITYTYPE_LIST, {"default": PERSONALITYTYPE_LIST[0]}),
                "Phobia": (PHOBIA_LIST, {"default": PHOBIA_LIST[0]}),
                "Occupation": (OCCUPATION_LIST, {"default": OCCUPATION_LIST[0]}),
                "Income": (INCOME_LIST, {"default": INCOME_LIST[0]}),
                "Sexual Orientation": (SEXUALORIENTATION_LIST, {"default": SEXUALORIENTATION_LIST[0]}),
                "Marital Status": (MARITALSTATUS_LIST, {"default": MARITALSTATUS_LIST[0]}),
                "Relation To Protagonist": (RELATIONTOPROTAGONIST_LIST, {"default": RELATIONTOPROTAGONIST_LIST[0]}),
                "First Appearance": (FIRSTAPPEARANCE_LIST, {"default": FIRSTAPPEARANCE_LIST[0]}),
                "Additional Details": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_info",)
    FUNCTION = "generate_character"
    CATEGORY = "triXope"

    def generate_character(self, **kwargs):
        if not kwargs.get("enabled", True):
            return ("",)
        name = kwargs.get("Name")
        age = kwargs.get("Age")
        race = kwargs.get("Race")
        role = kwargs.get("Role")
        eye_color = kwargs.get("Eye Color")
        skin_tone = kwargs.get("Skin Tone")
        body_type = kwargs.get("Body Type")
        posture = kwargs.get("Posture")
        face_shape = kwargs.get("Face Shape")
        hair_style = kwargs.get("Hair Style")
        hair_color = kwargs.get("Hair Color")
        wardrobe = kwargs.get("Wardrobe")
        life_skill = kwargs.get("Life Skill")
        life_strength = kwargs.get("Life Strength")
        hobby = kwargs.get("Hobby")
        musical_interest = kwargs.get("Musical Interest")
        bad_habit = kwargs.get("Bad Habit")
        faith = kwargs.get("Faith")
        pet = kwargs.get("Pet")
        personality_type = kwargs.get("Personality Type")
        phobia = kwargs.get("Phobia")
        occupation = kwargs.get("Occupation")
        income = kwargs.get("Income")
        sexual_orientation = kwargs.get("Sexual Orientation")
        marital_status = kwargs.get("Marital Status")
        relation_to_protagonist = kwargs.get("Relation To Protagonist")
        first_appearance = kwargs.get("First Appearance")
        details = kwargs.get("Additional Details")

        output_text = (
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Race: {race}\n"
            f"Role: {role}\n"
            f"Eye Color: {eye_color}\n"
            f"Skin Tone: {skin_tone}\n"
            f"Body Type: {body_type}\n"
            f"Posture: {posture}\n"
            f"Face Shape: {face_shape}\n"
            f"Hair Style: {hair_style}\n"
            f"Hair Color: {hair_color}\n"
            f"Wardrobe: {wardrobe}\n"
            f"Life Skill: {life_skill}\n"
            f"Life Strength: {life_strength}\n"
            f"Hobby: {hobby}\n"
            f"Musical Interest: {musical_interest}\n"
            f"Bad Habit: {bad_habit}\n"
            f"Faith: {faith}\n"
            f"Pet: {pet}\n"
            f"Personality Type: {personality_type}\n"
            f"Phobia: {phobia}\n"
            f"Occupation: {occupation}\n"
            f"Income: {income}\n"
            f"Sexual Orientation: {sexual_orientation}\n"
            f"Marital Status: {marital_status}\n"
            f"Relation To Protagonist: {relation_to_protagonist}\n"
            f"First Appearance: {first_appearance}\n"
            f"Additional Details: {details}"
        )
        
        return (output_text,)