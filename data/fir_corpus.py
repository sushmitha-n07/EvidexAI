# data/fir_corpus.py
# Sample FIR corpus for EvidexAI project
# Each entry contains an "intent" (crime type) and "text" (FIR narrative)

fir_corpus = [
    {
        "intent": "Domestic Violence",
        "text": "Victim was found unconscious in the kitchen. Suspected blunt force trauma. No signs of forced entry. Husband claims she fell."
    },
    {
        "intent": "Robbery",
        "text": "Victim's home broken into during night hours. Jewelry and cash missing. Broken window in living room. No witnesses."
    },
    {
        "intent": "Homicide",
        "text": "Male victim found dead in garage. Multiple stab wounds. Knife missing. Signs of struggle. No suspects identified yet."
    },
    {
        "intent": "Sexual Assault",
        "text": "Victim found in alley. Reports being followed. Minor injuries. Medical exam confirmed sexual assault. CCTV shows suspect wearing hoodie."
    },
    {
        "intent": "Accidental Death",
        "text": "Victim electrocuted while using hairdryer in bathroom. No foul play suspected. Wiring inspection requested."
    },
    {
        "intent": "Suicide",
        "text": "Note found near body. Hanging observed from ceiling fan. No signs of forced entry. Family unaware of mental health issues."
    },
    {
        "intent": "Cybercrime",
        "text": "Victim reported unauthorized access to bank account. Funds transferred overseas. Phishing email found in inbox."
    },
    {
        "intent": "Kidnapping",
        "text": "Minor reported missing from school premises. Witnesses saw unknown man leading child away. Ransom call received by parents."
    },
    {
        "intent": "Drug Offense",
        "text": "Police raid recovered illegal substances from suspect's apartment. Packets of heroin and cannabis seized. Suspect arrested."
    },
    {
        "intent": "Fraud",
        "text": "Victim reported being cheated in online transaction. Fake website collected payments. No goods delivered."
    },
    {
        "intent": "Terrorism",
        "text": "Explosive device found near railway station. Bomb squad neutralized. Suspected terror group claimed responsibility."
    },
    {
        "intent": "Human Trafficking",
        "text": "Multiple victims rescued from illegal confinement. Promised jobs abroad but forced into exploitation. Organized network suspected."
    }
]