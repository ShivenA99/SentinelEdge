"""Generate adversarial scam and hard-negative legitimate call transcripts.

Produces a CSV with columns: text, label, category.
Each row is a single sentence (label=1 for scam, label=0 for legit).

The adversarial scam examples are designed to evade the current detector by
avoiding obvious trigger words (urgency, threats, explicit org names) while
still performing social engineering.  Hard negatives are legitimate calls that
*look* suspicious (real fraud alerts, real IRS refund calls, etc.) to stress-
test the false-positive rate.

Categories
----------
Scam (label=1):
  - polite_social_engineering      : friendly tone, no urgency, no threats
  - gradual_escalation             : starts as normal chat, slowly introduces scam
  - vague_impersonation            : impersonates authority without naming orgs
  - emotional_manipulation         : grandparent/romance/charity variants
  - technical_sophistication       : fake IT/security with real jargon
  - legitimate_sounding_scam       : fake appointments/deliveries/refunds

Hard negative (label=0):
  - real_fraud_alert               : genuine bank fraud department
  - real_government_contact        : real IRS/SSA outreach
  - real_tech_support              : real product support
  - real_urgent_legitimate         : doctor, school, emergency

Usage
-----
    python training/generate_adversarial_data.py \
        [--output data/raw/synthetic_transcripts/adversarial_calls.csv] \
        [--seed 123]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

# ---------------------------------------------------------------------------
# Shared fill helpers  (reuse same helper style as generate_synthetic_data.py)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony",
    "Mark", "Steven", "Andrew", "Paul", "Joshua", "Kenneth",
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Emily",
    "Margaret", "Ashley", "Kimberly", "Donna", "Sandra",
    "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Lopez", "Wilson", "Anderson",
    "Taylor", "Moore", "Jackson", "Martin", "Lee",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "San Antonio", "Dallas", "Miami", "Atlanta", "Denver",
    "Seattle", "Boston", "Portland", "Nashville", "Austin",
]

AMOUNTS = [
    "$49.99", "$89", "$129", "$199", "$250", "$349", "$499", "$750",
    "$1,200", "$2,500", "$4,800", "$7,500",
]

SMALL_AMOUNTS = ["$4.99", "$9.99", "$14.95", "$19.99", "$29.95", "$39.99"]

DATES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "next Monday", "next Friday", "tomorrow", "March 15th", "April 2nd",
    "this coming Thursday", "the 20th of this month",
]

TIMES = [
    "9:00 AM", "10:15 AM", "11:30 AM", "1:00 PM", "2:30 PM", "3:45 PM",
]

COMPANY_NAMES = [
    "Pinnacle Solutions", "Vertex Industries", "Horizon Tech",
    "Summit Global", "Apex Consulting", "Meridian Partners",
    "BlueStar Enterprises", "Nexus Group",
]

PRODUCT_NAMES = [
    "wireless speaker", "laptop stand", "office chair", "printer",
    "monitor", "keyboard", "router", "tablet", "external drive", "webcam",
]

CHILD_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Logan", "Mia", "Lucas", "Charlotte", "Aiden",
]

PHONE_NUMBERS = [
    f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"
    for _ in range(50)
]

LAST4 = [str(random.randint(1000, 9999)) for _ in range(50)]

ACCOUNT_SUFFIXES = [str(random.randint(1000, 9999)) for _ in range(50)]


def _fill(template: str) -> str:
    """Replace {placeholder} tokens in a template with random values."""
    replacements = {
        "{name}": random.choice(FIRST_NAMES),
        "{caller}": random.choice(FIRST_NAMES),
        "{city}": random.choice(CITIES),
        "{amount}": random.choice(AMOUNTS),
        "{small_amount}": random.choice(SMALL_AMOUNTS),
        "{date}": random.choice(DATES),
        "{time}": random.choice(TIMES),
        "{company}": random.choice(COMPANY_NAMES),
        "{product}": random.choice(PRODUCT_NAMES),
        "{child}": random.choice(CHILD_NAMES),
        "{phone}": random.choice(PHONE_NUMBERS),
        "{last4}": random.choice(LAST4),
        "{acct_suffix}": random.choice(ACCOUNT_SUFFIXES),
        "{years}": str(random.randint(3, 15)),
        "{ref}": f"REF-{random.randint(100000,999999)}",
        "{ticket}": f"TK-{random.randint(10000,99999)}",
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


# ======================================================================
# CATEGORY 1 : Polite Social Engineering  (label=1)
# No urgency, no threats, friendly tone -- but asks for sensitive info.
# ======================================================================

POLITE_SOCIAL_ENGINEERING = {
    "openers": [
        "Hi there, I am calling from your financial institution's loyalty program.",
        "Good {time}, this is {name} from the rewards division of your service provider.",
        "Hello, hope you are having a great day, I am reaching out from our member benefits team.",
        "Hi, this is {name} calling on behalf of your card issuer's customer appreciation department.",
        "Good {time}, we noticed you have been a valued member for {years} years and we would like to say thank you.",
        "Hello, I am {name} from the premium services group, and I have some good news for you.",
        "Hi there, just a quick courtesy call from our retention department to share some exciting updates.",
        "Good afternoon, I am calling because you have been selected for a complimentary service upgrade.",
        "Hello, this is {name} and I am reaching out on behalf of your plan administrator with a small gift for loyal members.",
        "Hi, we are doing a brief check-in with our long-standing customers to make sure you are getting the most from your plan.",
        "Good morning, I am {name} and your account has been flagged for a positive review.",
        "Hello, I hope I am not interrupting, this is {name} from the customer loyalty team.",
        "Hi there, we have a special offer exclusively for customers like you who have maintained excellent standing.",
        "Good {time}, this is a friendly call to let you know about a free benefit enhancement available to you.",
        "Hello, I am calling with good news, your long membership qualifies you for a complimentary upgrade.",
    ],
    "builds": [
        "We would love to apply a loyalty bonus to your profile, it only takes a moment.",
        "There is no cost at all, this is purely a thank-you for your continued membership.",
        "We have already pre-approved the upgrade on our end, I just need to match it to the right profile.",
        "The whole process is very straightforward and should take less than two minutes.",
        "A lot of our members have taken advantage of this and they have been very pleased.",
        "I completely understand if you have questions, I am happy to walk you through everything.",
        "This benefit includes a reduced rate on your next billing cycle, which I think you will appreciate.",
        "There is absolutely no obligation, and you can cancel the upgrade at any time.",
        "We are reaching out proactively so you do not miss out on something you have already earned.",
        "I want to make sure this goes smoothly, so I will guide you through each step.",
        "Many of our premium members have told us this was one of the best perks they received.",
        "Our goal is just to make your experience better, nothing more.",
        "Everything is handled on our end, I just need to confirm a couple of details.",
        "You deserve the best service, and this enhancement reflects that commitment.",
        "I promise this will be quick and painless, and you will see the benefit right away.",
    ],
    "extractions": [
        "Could you just confirm the account number so I can apply the benefit to the correct profile?",
        "I just need to verify your date of birth and the last four of your identification number to pull up your record.",
        "Can you read me the number on the front of your card so I can locate your membership?",
        "To proceed I will need your full name as it appears on the account and your mailing address.",
        "For security purposes, could you share the three-digit code on the back of your card?",
        "Let me just verify your identity with your mother's maiden name and zip code.",
        "Could you confirm your login email and the password hint so I can pull up the right account?",
        "I need the routing number associated with your primary account for the direct credit.",
        "May I have your full card number and expiration date so the system can match your profile?",
        "Just to make sure I have the right person, could you tell me your Social Security number?",
        "Would you mind sharing your online banking username so I can link the reward?",
        "I will need the PIN you use at the ATM to verify the card on file.",
        "Can you provide the security code that was just sent to your phone to verify the account?",
        "Please share the full account number including the branch code for the reward credit.",
        "To finalize, I will need you to read back the one-time code that just arrived via text.",
    ],
    "closers": [
        "Thank you so much for your time, you should see the benefit reflected within a few days.",
        "It has been a pleasure speaking with you, enjoy the upgrade.",
        "We appreciate your loyalty and hope you continue to enjoy your membership.",
        "If you have any questions later, feel free to call the number on your statement.",
        "Thank you and have a wonderful rest of your day.",
    ],
}

# ======================================================================
# CATEGORY 2 : Gradual Escalation  (label=1)
# Starts as completely normal chat, slowly introduces scam elements.
# ======================================================================

GRADUAL_ESCALATION = {
    "innocent_start": [
        "Hey, how are you doing today?",
        "Hi there, hope you are having a good week so far.",
        "Good {time}, beautiful day out there, isn't it?",
        "Hello, I was just thinking about you the other day.",
        "Hi, it has been a while since we chatted, how is everything?",
        "Hey, did you catch the game last night?",
        "Hi there, I just had the best coffee this morning, have you tried that new place downtown?",
        "Good {time}, I heard the weather is supposed to be nice this weekend.",
        "Hello, I saw your post about your garden, it looks amazing this year.",
        "Hi, hope the family is doing well, how are the kids?",
        "Hey, I just got back from a trip to {city}, it was fantastic.",
        "Good morning, I have been meaning to reach out for a while now.",
        "Hi there, congratulations on the new job, I heard through the grapevine.",
        "Hello, I bumped into {name} at the store and they mentioned you.",
        "Hey, remember that restaurant we talked about, I finally tried it and it was great.",
    ],
    "normal_middle": [
        "So I have been keeping busy with work, the usual stuff.",
        "Things have been pretty quiet on my end, just the day-to-day routine.",
        "I have been spending more time outdoors lately, trying to stay active.",
        "The kids are growing up so fast, {child} just started a new school.",
        "We are thinking about doing some home renovations this summer.",
        "I picked up a new hobby recently, been learning to cook different cuisines.",
        "Have you been watching any good shows lately? I need recommendations.",
        "The traffic has been terrible around here, they are doing road work everywhere.",
        "I tried that new recipe you mentioned and it turned out pretty decent.",
        "We just adopted a cat, the whole family is thrilled about it.",
        "My neighbor just put their house on the market, things are changing around here.",
        "I signed up for a yoga class, trying to be healthier this year.",
        "The local farmers market started up again, the produce has been excellent.",
        "I have been reading a lot more lately, just finished a great novel.",
        "We hosted a small get-together last weekend, it was a lot of fun.",
    ],
    "transition": [
        "Oh by the way, something a little odd happened with my statement recently.",
        "Speaking of which, I noticed something unusual when I checked my account online.",
        "That reminds me, I got a strange notification about my profile the other day.",
        "On a different note, I was looking at my records and something did not look right.",
        "Funny you should mention finances, I had an issue come up with my billing.",
        "While I have you, have you ever had a problem with your account being accessed?",
        "Actually, now that I think about it, there was something I wanted to ask you about.",
        "This might sound random, but have you been getting any weird charges on your card?",
        "You know what, something came up this week that I could use your advice on.",
        "On an unrelated topic, did you know there is a way to check if your information has been exposed?",
        "Actually I almost forgot, I wanted to mention something that happened to me.",
        "Changing the subject a bit, have you reviewed your statements recently?",
        "Incidentally, I ran across a situation that might affect people like us.",
        "Before I forget, there is something I learned about recently that might interest you.",
        "So I do not want to alarm you, but I heard something that made me think of you.",
    ],
    "scam_reveal": [
        "I found out there is a way to get a refund on overpayments, but they need your account details to process it.",
        "Apparently if you share your card number with this service they can run a free security audit.",
        "They told me that to opt out of the data breach settlement you need to provide your Social Security number.",
        "I was told that if you call this number and give them your PIN they can check if your card has been cloned.",
        "There is a website where you enter your banking credentials and it scans for unauthorized access.",
        "My friend said they got money back by providing their account and routing number to a verification service.",
        "They said all you need to do is share your login details and they run a complimentary security check.",
        "I heard you can get a free credit monitoring service if you just provide your identification number.",
        "Someone told me that if you wire a small deposit to this escrow service they double it as a promotional offer.",
        "The trick is to give them the one-time code from your phone and they unlock the premium tier for free.",
        "All they need is the security code on the back of your card to run the background verification.",
        "They are offering a guaranteed return on a small transfer, my neighbor said she already received her payout.",
        "If you send a small processing fee they will release the full refund to your account.",
        "I was advised to provide my full card details to verify eligibility for the rebate program.",
        "The representative said they need your date of birth and last four digits to confirm the payout.",
    ],
    "closers": [
        "Anyway, just thought I would pass that along, hope it helps.",
        "Let me know if you try it, I would love to hear how it goes.",
        "Alright, I should get going, but think about what I said.",
        "Talk soon, and seriously consider looking into that.",
        "Take care, and do not wait too long on that opportunity.",
    ],
}

# ======================================================================
# CATEGORY 3 : Vague Impersonation  (label=1)
# Impersonates authority without naming specific organizations.
# ======================================================================

VAGUE_IMPERSONATION = {
    "openers": [
        "Hello, I am calling from your financial institution regarding your account.",
        "Good {time}, this is a representative from the government agency that oversees your benefits.",
        "Hi, I am reaching out from the office that handles your retirement contributions.",
        "Hello, this is {name} from the division responsible for your tax filing records.",
        "Good {time}, I am calling from the entity that manages your health coverage.",
        "Hi there, I represent the bureau that processes identification documents in your state.",
        "Hello, I am with the department that administers your student loan portfolio.",
        "Good afternoon, this is {name} from the regulatory body overseeing consumer protections in your area.",
        "Hi, I am calling on behalf of the organization that handles your payroll direct deposits.",
        "Hello, I represent the office responsible for your property tax assessment.",
        "Good {time}, this is the compliance division for the institution where you hold your primary account.",
        "Hi, I am from the agency that coordinates disaster relief in your region.",
        "Hello, this is {name} from the oversight committee reviewing accounts in your district.",
        "Good morning, I am contacting you from the administration that manages public records for your county.",
        "Hi there, I am with the group responsible for resolving billing discrepancies on your file.",
    ],
    "builds": [
        "We have been conducting a routine review and your file came up for attention.",
        "There is nothing to worry about, this is standard procedure that we do periodically.",
        "Our records indicate there may be a minor discrepancy that we would like to clear up.",
        "We are updating our systems and want to make sure your information is current.",
        "A recent policy change requires us to re-verify certain details for all account holders.",
        "This is part of a broader initiative to improve service quality for everyone.",
        "We want to make sure you are receiving all the benefits you are entitled to.",
        "Our team identified a small issue during an audit that we can resolve right now.",
        "There has been a system migration and we are reaching out to confirm data integrity.",
        "We are contacting a select group of individuals to offer an updated service tier.",
        "Your file was flagged during a routine check, and it is probably nothing serious.",
        "We noticed you may be eligible for an adjustment that could work in your favor.",
        "As part of our annual reconciliation, we need to touch base with certain account holders.",
        "A new regulation went into effect and we need to update our records accordingly.",
        "This is a proactive outreach to ensure everything is in order before the end of the fiscal period.",
    ],
    "extractions": [
        "I just need to confirm your full identification number to access the correct file.",
        "Could you provide the account number associated with your primary financial relationship?",
        "For verification purposes, I will need your date of birth and the last four of your identification.",
        "Can you share the login credentials for the online portal so I can review the record with you?",
        "I need the routing and account numbers for the institution that handles your deposits.",
        "Would you mind confirming the three-digit security code associated with your primary card?",
        "To proceed, I will need the passcode that was sent to your registered device.",
        "Please share the full sixteen-digit number on the front of your card for identification.",
        "I need your complete mailing address and the mother's maiden name on file.",
        "Could you verify the PIN number associated with your account for our records?",
        "I will need the access code you use for telephone banking to locate your profile.",
        "Can you provide the security answers for the questions listed on your account?",
        "For the update I need your bank statement password so I can pull the correct records.",
        "Please confirm your employer identification number so we can cross-reference your file.",
        "I just need the one-time verification code that was just sent to your mobile number.",
    ],
    "closers": [
        "Thank you for your cooperation, this should be all resolved now.",
        "We appreciate your time, your records are now up to date.",
        "If you have any further questions, please call the main number during business hours.",
        "Everything looks good on our end, you should not need to do anything else.",
        "Thank you, and we apologize for any inconvenience, have a great day.",
    ],
}

# ======================================================================
# CATEGORY 4 : Emotional Manipulation  (label=1)
# Grandparent variants, romance scam, charity scam.
# ======================================================================

EMOTIONAL_MANIPULATION = {
    "grandparent_soft": [
        "Hi, it is me, I did not want to worry you but I have gotten into a bit of a situation.",
        "Hey, I know it has been a while since I called, but I could really use your help with something.",
        "Hi, I am sorry to bother you, I just did not know who else to turn to.",
        "I feel terrible asking, but you have always been the one person I could count on.",
        "I promise this is not a big deal, but I need a small favor and I will pay you right back.",
        "You are the only one I trust enough to ask about this, please do not mention it to anyone.",
        "I hate to put this on you, but something came up and I am in a tough spot.",
        "I know this sounds unusual but please hear me out before you say anything.",
        "I did not want to involve anyone else because I am a little embarrassed about the whole thing.",
        "Please just listen, I am okay but I need some help getting through a rough patch.",
        "I am fine, really, it is just a minor thing but I need a bit of financial help.",
        "You know I would never ask unless it was really important, and I will explain everything later.",
        "I am somewhere unfamiliar and I just need you to help me sort something out.",
        "Please do not be upset, everything is going to be fine, I just need a little assistance.",
        "I tried handling this on my own but it did not work out, so I am reaching out to you.",
    ],
    "romance_build": [
        "I have been thinking about you all day and I feel like we have a real connection.",
        "I know we have only been talking for a little while, but I feel like I have known you forever.",
        "You are the most understanding person I have ever met and I treasure our conversations.",
        "I want to be honest with you because you mean so much to me.",
        "I have never felt this way about anyone and I hope you feel the same.",
        "Being able to talk to you is the best part of my day, honestly.",
        "I wish I could be there with you right now, this distance is so hard.",
        "You are the first person I think about when I wake up and the last before I sleep.",
        "I know this is all happening fast, but sometimes when you know, you know.",
        "I dream about the day we can finally meet in person and start our life together.",
        "Everything about you makes me happy, your kindness, your humor, your warmth.",
        "I have been saving up to come visit you, it is all I can think about.",
        "You are the only person who truly understands me.",
        "I have never trusted anyone the way I trust you.",
        "I feel so lucky to have found you, you are my everything.",
    ],
    "romance_ask": [
        "Something unexpected happened and I need a little financial help to get through it.",
        "My account was frozen due to a mix-up and I just need a small loan until it is sorted out.",
        "I got stuck with an unexpected medical expense and I am short by just a little bit.",
        "I was about to book my flight to come see you but my card was declined unexpectedly.",
        "I need to pay for a visa application to come to your country and I am just a bit short.",
        "The hospital is asking for a deposit before they can treat me and I do not have enough.",
        "I lost my wallet while traveling and I need help covering my hotel and food until I get home.",
        "There is a fee to release my package from customs and I was not expecting it.",
        "I need help paying for a background check they require before I can enter your country.",
        "My landlord is threatening to change the locks unless I pay rent by end of day.",
        "I had an accident and the repair costs are more than I can handle right now.",
        "They are asking for a processing fee to transfer my savings to an international account.",
        "I need a small amount to cover legal paperwork so we can be together sooner.",
        "My phone plan was cut off and I need to pay the balance to keep talking to you.",
        "I just need a wire transfer to cover the travel insurance required for my trip to see you.",
    ],
    "charity_appeal": [
        "We are a small group of volunteers helping families affected by the recent disaster.",
        "I am calling to see if you might be able to make a contribution to help displaced families.",
        "Every little bit helps, even a small donation can make a real difference for these children.",
        "We have been working around the clock to provide food and shelter to those in need.",
        "The situation on the ground is heartbreaking and any support is deeply appreciated.",
        "We are not a big organization, just neighbors helping neighbors in a time of crisis.",
        "I have personally seen the impact of the donations we collect and it gives people hope.",
        "The families we serve have lost everything and they need immediate help.",
        "We have been overwhelmed by the response from caring people like you.",
        "Your generosity could help a family get back on their feet after losing their home.",
        "We focus on getting supplies directly to the people who need them most.",
        "Even a small amount will go a long way, we operate with almost zero overhead.",
        "I am volunteering my time because I saw the devastation firsthand and had to do something.",
        "We are trying to raise enough to keep the shelter open through the end of the month.",
        "These are real people in your community who need help right now.",
    ],
    "charity_extract": [
        "We accept donations through wire transfer, can I get your bank details to set that up?",
        "The easiest way to contribute is to share your card number over the phone right now.",
        "If you could send a gift card with the codes, we can convert that to supplies immediately.",
        "We have a special fund where you can send money directly via a personal transfer.",
        "Could you provide your payment information so we can process the donation securely?",
        "A wire transfer is the fastest method, I can give you the account details to send it to.",
        "We prefer donations via money transfer services as they have the lowest fees.",
        "If you have a prepaid card, the codes work just as well as a direct deposit for us.",
        "Can I take your credit card details now so the donation goes through before the deadline?",
        "We set up a special account at a transfer service, could you send the funds there?",
        "Please share your banking details and we will process a one-time deduction.",
        "The quickest way to help is to send us a payment through a peer-to-peer app.",
        "Could you purchase store gift cards and share the numbers, we redeem them for supplies.",
        "We are collecting through a digital wallet, I can give you the address to transfer to.",
        "If you share your card details I can run the donation right now while we are on the phone.",
    ],
    "closers": [
        "Thank you so much, you have no idea how much this means.",
        "I really appreciate you, I will make it up to you, I promise.",
        "You are a lifesaver, I do not know what I would do without you.",
        "Please keep this between us for now, I will explain everything soon.",
        "Thank you from the bottom of my heart, you are making a real difference.",
    ],
}

# ======================================================================
# CATEGORY 5 : Technical Sophistication  (label=1)
# Fake IT department with real-sounding jargon, no panic language.
# ======================================================================

TECHNICAL_SOPHISTICATION = {
    "openers": [
        "Hi, this is {name} from the IT helpdesk, we are performing a scheduled security review.",
        "Hello, I am calling from the information security team about a routine credential rotation.",
        "Good {time}, this is {name} from network operations, we are updating our encryption protocols.",
        "Hi there, I am from the infrastructure team and we are rolling out a mandatory firmware patch.",
        "Hello, this is the identity and access management group, we are conducting a periodic access audit.",
        "Good {time}, I am {name} from the cybersecurity compliance team following up on a configuration change.",
        "Hi, I am reaching out from the systems administration team regarding a certificate renewal.",
        "Hello, this is {name} with the endpoint security division, we need to re-authenticate your device.",
        "Good afternoon, the IT governance committee has asked us to verify access credentials for all staff.",
        "Hi, I am calling from the security operations center about a planned migration to our new authentication platform.",
        "Hello, this is {name} from IT, we are upgrading our multi-factor authentication system this week.",
        "Good {time}, I am from the IT compliance office, we are conducting our quarterly access review.",
        "Hi there, this is the help desk following up on ticket {ticket} regarding your account settings.",
        "Hello, I am {name} from digital workspace services, we noticed your profile needs a certificate update.",
        "Good {time}, this is IT services calling about the organization-wide password policy change.",
    ],
    "builds": [
        "We are migrating all user accounts to a new LDAP directory and need to validate each profile.",
        "As part of our zero-trust implementation, we are re-issuing access tokens for all employees.",
        "The new TLS 1.3 configuration requires us to re-authenticate each endpoint individually.",
        "Our SIEM platform flagged your account during a routine scan, it is most likely a false positive.",
        "We are deploying a new PKI infrastructure and every user needs to renew their digital certificate.",
        "The SOC team detected an anomalous login pattern that we want to rule out as benign.",
        "This is standard procedure whenever we update the Kerberos authentication backend.",
        "We are replacing the existing VPN concentrator and need to re-provision client configurations.",
        "The NIST compliance framework requires annual re-verification of all privileged accounts.",
        "Our intrusion detection system picked up some unusual network traffic from your subnet.",
        "We are hardening our Active Directory environment as part of the new security baseline.",
        "The certificate authority we use is updating its root certificates and all clients must re-enroll.",
        "Our vulnerability scanner identified a misconfiguration that requires a credential refresh.",
        "We are implementing conditional access policies and need to verify your device posture.",
        "The change advisory board approved the update and we are now reaching out to affected users.",
    ],
    "extractions": [
        "I will need your current network password so we can reset it to the new standard.",
        "Could you provide your VPN credentials so I can test the new tunnel configuration?",
        "Please share the six-digit code from your authenticator app so I can bind it to the new system.",
        "I need your Active Directory username and password to complete the migration.",
        "Can you give me remote access to your machine so I can run the configuration script?",
        "I will need you to navigate to a URL I provide and enter your credentials for the new portal.",
        "Please share the recovery codes for your multi-factor authentication setup.",
        "I need the passphrase for your encrypted drive so we can update the recovery key.",
        "Could you read me the verification code that just appeared on your authentication app?",
        "I will need your SSH private key or at least the passphrase to regenerate a new key pair.",
        "Please provide the admin password for your local machine so we can push the update.",
        "Can you share your single sign-on credentials so I can verify the integration?",
        "I need the API token from your developer account to migrate it to the new platform.",
        "Please enter your password on the screen I am about to share with you.",
        "I will need your email account password to reconfigure the mail client with the new certificates.",
    ],
    "closers": [
        "That should be everything, your account will be fully migrated within the hour.",
        "Thanks for your cooperation, we are trying to get everyone updated before the maintenance window.",
        "If you experience any login issues, just reach out to the help desk and reference this ticket.",
        "We appreciate your patience, security updates like this help keep everyone protected.",
        "All set on our end, you should not notice any disruption to your normal workflow.",
    ],
}

# ======================================================================
# CATEGORY 6 : Legitimate-Sounding Scam  (label=1)
# Fake appointments, deliveries, refunds that extract sensitive info.
# ======================================================================

LEGITIMATE_SOUNDING_SCAM = {
    "fake_appointment": [
        "Hello, this is {name} from the medical records department, we are updating your file before your visit.",
        "Hi, I am calling to confirm your upcoming procedure and need to verify some details.",
        "Good {time}, our billing team needs your insurance information updated before your appointment on {date}.",
        "Hello, to finalize your appointment we need your Social Security number for the insurance pre-authorization.",
        "Hi, the doctor's office asked me to collect your payment card details for the copay before you arrive.",
        "We noticed your insurance information on file may be outdated, could you read me the policy number and your SSN?",
        "Good {time}, this is the scheduling team, we need your date of birth and the last four of your Social to confirm identity.",
        "Hello, your appointment requires a pre-payment of {small_amount} to hold the slot, may I have your card number?",
        "Hi, the pharmacy needs your card on file to process a small balance before your prescription is ready.",
        "Good afternoon, we are verifying patient records and need you to confirm your full identification number.",
        "Hello, this is the front desk calling ahead, we need your full legal name and identification to match our records.",
        "Hi, the insurance coordinator needs your bank routing number to process the claim refund for your last visit.",
        "Good morning, we are calling about your lab results and need to verify your identity with your full Social Security number.",
        "Hello, our records show an unpaid balance of {small_amount} from your last visit, could I take a card number?",
        "Hi, we are updating our electronic health records and need your full name, date of birth, and Social Security to merge your files.",
    ],
    "fake_delivery": [
        "Hello, this is the shipping department, we have a package for you but need to confirm your identity.",
        "Hi, your delivery was flagged by our system and we need a small re-delivery fee of {small_amount}.",
        "Good {time}, there is a customs charge of {small_amount} on your international package, can I take a card number?",
        "Hello, your item has arrived at our facility but we need your full name and identification to release it.",
        "Hi, the carrier requires your date of birth and the last four of your Social to verify the recipient.",
        "We attempted delivery but need you to confirm your billing address and payment details for the reshipping fee.",
        "Good {time}, your package is on hold, to release it we need the credit card used for the original purchase.",
        "Hello, a signature and payment of {small_amount} are required before we can complete delivery.",
        "Hi, our records show an incomplete address, could you provide your full details including your account number?",
        "Good morning, we need to run a security verification before releasing your high-value package.",
        "Hello, your package requires additional insurance and we need a card number to process the {small_amount} charge.",
        "Hi, the delivery driver reported a discrepancy, I need your identification number to resolve it.",
        "Good afternoon, there is a surcharge of {small_amount} for Saturday delivery, may I have your payment details?",
        "Hello, to reschedule your delivery we need to verify your identity with your date of birth and ID number.",
        "Hi, your shipment is at the distribution center, a handling fee of {small_amount} is required for release.",
    ],
    "fake_refund": [
        "Hello, this is the refund processing department, we owe you {amount} from an overcharge on your account.",
        "Hi, I am calling because a billing error resulted in a double charge and we need to return the funds.",
        "Good {time}, you are entitled to a refund of {amount} but I need your bank details to process the transfer.",
        "Hello, our system overcharged you by {amount} last month and we want to make it right.",
        "Hi, to send the refund of {amount} directly to your account, I will need the routing and account numbers.",
        "We discovered a pricing error that affected your last three transactions and you are owed {amount}.",
        "Good {time}, the refund has been approved on our end, I just need your card details to push it through.",
        "Hello, the credit of {amount} is ready to be applied, could you verify your card number and expiration?",
        "Hi, our accounting team flagged your account for an overpayment, we would like to return {amount} to you.",
        "Good afternoon, we processed your refund request and need your banking information to complete the transfer.",
        "Hello, a price adjustment of {amount} is owed to you, I need the full card number to apply the credit.",
        "Hi, to expedite your refund I will need the login credentials for your online banking portal.",
        "Good morning, the refund of {amount} requires your PIN to verify it goes to the right card.",
        "Hello, our billing department confirmed you are owed {amount}, may I have your account and routing numbers?",
        "Hi, to reverse the charge of {amount} I need the three-digit code on the back of your card.",
    ],
    "closers": [
        "Thank you, the refund should appear within three to five business days.",
        "We appreciate your patience, everything should be taken care of now.",
        "If you have any questions, feel free to reach out during business hours.",
        "Thank you for verifying, your package will be out for delivery shortly.",
        "We are sorry for the inconvenience and appreciate your understanding.",
    ],
}

# ======================================================================
# HARD NEGATIVES  (label=0)
# Legitimate calls that look suspicious but are genuine.
# ======================================================================

HARD_NEGATIVE_REAL_FRAUD_ALERT = {
    "openers": [
        "Hello, this is the fraud monitoring team at your bank, we detected a suspicious charge on your card ending in {last4}.",
        "Hi, this is {name} from your credit card company's fraud department, I need to discuss some recent activity.",
        "Good {time}, we flagged a transaction on your account and want to verify whether it was authorized.",
        "Hello, our automated system detected an unusual purchase and I am calling to confirm it with you.",
        "Hi, this is the security team at your financial institution, there was an attempted charge in {city} on your card.",
        "Good afternoon, this is {name} from fraud prevention, a charge of {amount} was flagged on your account.",
        "Hello, we are calling because your card was used at a location that does not match your spending history.",
        "Hi, our fraud detection system triggered an alert and we want to make sure your card is secure.",
        "Good {time}, this is the fraud department, we are going to ask you some yes or no questions about recent transactions.",
        "Hello, a purchase attempt was made online for {amount} and we temporarily blocked it pending your confirmation.",
        "Hi, this is your bank calling about a potentially unauthorized ATM withdrawal in {city}.",
        "Good morning, our system noticed multiple rapid transactions on your card which is unusual for your account.",
        "Hello, this is {name} from card security, we need to verify a charge before we release the hold.",
        "Hi, we blocked a wire transfer attempt from your account and need to confirm if you initiated it.",
        "Good {time}, this is the real-time fraud team, a charge of {amount} at an electronics store was flagged.",
    ],
    "middles": [
        "I am not going to ask for your full card number or any passwords, I just need you to confirm or deny recent charges.",
        "For your security, we will never ask for your PIN or online banking password during this call.",
        "I can see the last four digits of your card here, can you confirm a charge of {amount} at a gas station in {city}?",
        "We have temporarily frozen the card to prevent further charges while we sort this out.",
        "You can verify this call is legitimate by calling the number on the back of your card.",
        "We are going to read off a few transactions and you just tell us if they look familiar.",
        "If any of these charges are not yours, we will reverse them and send you a new card.",
        "The hold on your card is temporary and will be lifted once we complete this review.",
        "We want to make sure no one else has access to your card information.",
        "A new card will be mailed to your address on file if we determine fraud occurred.",
        "Please check your email for an official communication from us with a reference number.",
        "We recommend changing your online banking password as a precaution after this review.",
        "Our investigation team will follow up within 48 hours if additional action is needed.",
        "You will not be held responsible for any unauthorized charges on your account.",
        "We take these matters very seriously and appreciate your help in resolving this quickly.",
    ],
    "closers": [
        "Thank you for confirming, we will issue a replacement card within five to seven business days.",
        "If you notice anything else unusual, please call us at the number on your card.",
        "We appreciate your cooperation, your account is now secure.",
        "A written summary of this fraud claim will be mailed to your address on file.",
        "Thank you for your time, and remember you can always verify by calling us directly.",
    ],
}

HARD_NEGATIVE_REAL_GOVERNMENT = {
    "openers": [
        "Hello, this is {name} from the Internal Revenue Service, we are calling about a refund on your most recent tax return.",
        "Hi, I am reaching out from the Social Security Administration regarding a change in your benefit amount.",
        "Good {time}, this is {name} with the IRS taxpayer assistance line, you requested a callback about your filing.",
        "Hello, we are contacting you from the Department of Motor Vehicles about your license renewal.",
        "Hi, this is the county tax assessor's office calling about your property tax reassessment.",
        "Good {time}, I am {name} from Medicare services, your coverage options are changing and we want to inform you.",
        "Hello, the IRS received your correspondence and I am calling to discuss the adjustment on your return.",
        "Hi, this is the Veterans Administration calling about a change in your disability rating.",
        "Good afternoon, this is {name} from Social Security, we are confirming the direct deposit change you requested.",
        "Hello, the state unemployment office is calling to verify your weekly certification details.",
        "Hi, this is {name} from the IRS resolution unit, we are following up on the installment agreement you set up.",
        "Good {time}, the Department of Education is reaching out about changes to your student loan repayment plan.",
        "Hello, we are calling from the county clerk's office about a document you submitted for recording.",
        "Hi, the Social Security office has received your application for survivors benefits and has a few questions.",
        "Good morning, this is the state health exchange calling about your insurance enrollment confirmation.",
    ],
    "middles": [
        "We already have your information on file, I am not going to ask for your Social Security number.",
        "You can verify this call by contacting our main office number listed on the official government website.",
        "The refund amount of {amount} will be deposited to the bank account associated with your last filing.",
        "We just need you to confirm the mailing address we have on file so we can send the notice.",
        "There is no payment required, this is purely informational about a change to your account.",
        "Our records show your benefits will increase by a small amount starting next month.",
        "This call is being made because you contacted us first and requested follow-up information.",
        "We encourage you to review the letter we sent and call us if anything does not match your records.",
        "Your case has been assigned reference number {ref} which you can use for any future inquiries.",
        "No action is required on your part unless you disagree with the adjustment.",
        "We never require payment over the phone for government services, please be aware of scams.",
        "Please allow ten to fourteen business days for the updated documents to arrive by mail.",
        "If you prefer, you can handle this in person at your local office by appointment.",
        "We want to make sure you are receiving everything you are entitled to under the current guidelines.",
        "The adjustment is retroactive and should reflect in your next payment cycle.",
    ],
    "closers": [
        "Thank you for your time, please do not hesitate to reach out if you have questions.",
        "You will receive a confirmation letter in the mail within two weeks.",
        "If you have concerns about this call, please contact us directly through the number on our official website.",
        "We appreciate your patience and are happy to help if you need anything further.",
        "Have a good day, and remember, no government agency will ask you to pay with gift cards.",
    ],
}

HARD_NEGATIVE_REAL_TECH_SUPPORT = {
    "openers": [
        "Hello, this is {name} from {company} product support, you submitted a service request for your {product}.",
        "Hi, I am calling back from technical support regarding the issue you reported with your {product}.",
        "Good {time}, this is {name} with {company} warranty service, your repair ticket has been updated.",
        "Hello, thank you for contacting us about your {product}, I have some information about your case.",
        "Hi, this is the customer care team at {company}, we are following up on your recent support inquiry.",
        "Good {time}, I am {name} from {company}, you contacted us about a billing question on your subscription.",
        "Hello, this is technical support returning your call about connectivity issues with your {product}.",
        "Hi, I am reaching out from {company} because you scheduled a callback for help with your account setup.",
        "Good afternoon, this is {name} from {company}, we want to update you on the repair status of your {product}.",
        "Hello, we received your email about the defective {product} and I am calling to arrange a replacement.",
        "Hi, this is {name} at {company}, your {product} is ready for pickup after the repair.",
        "Good {time}, I am from the {company} help desk, you opened a ticket about your software license.",
        "Hello, this is {name} from {company}, we have a firmware update available for your {product}.",
        "Hi, I am calling because you reported an issue with your {product} and I have some troubleshooting steps.",
        "Good morning, this is {company} support, your replacement {product} has shipped and I have tracking info.",
    ],
    "middles": [
        "I can walk you through the troubleshooting steps if you have a few minutes now.",
        "Your warranty covers this repair at no additional cost to you.",
        "The replacement part has been ordered and should arrive within three business days.",
        "I am going to send you an email with step-by-step instructions so you can reference them later.",
        "If the issue persists after these steps, we can arrange for a technician to visit.",
        "We identified the root cause as a firmware bug and the update should resolve it.",
        "Your case has been escalated to our engineering team for further investigation.",
        "I do not need any payment information for this service call, it is covered under your plan.",
        "You can also find self-help resources on our support website if you prefer.",
        "We are sending a prepaid shipping label so you can return the defective unit at no charge.",
        "The software update is available now and takes about ten minutes to install.",
        "Our technicians tested a replacement unit and confirmed it resolves the reported issue.",
        "I will keep your ticket open until we confirm the issue is fully resolved.",
        "If you experience the problem again, you can reference ticket number {ticket}.",
        "We value your feedback and will use it to improve the product in future releases.",
    ],
    "closers": [
        "Is there anything else I can help you with regarding your {product}?",
        "Thank you for being a {company} customer, we are happy to help.",
        "You will receive a follow-up email summarizing everything we discussed today.",
        "If you need further assistance, do not hesitate to contact us again.",
        "We hope this resolves the issue, have a great rest of your day.",
    ],
}

HARD_NEGATIVE_REAL_URGENT = {
    "openers": [
        "Hello, this is Dr. {name}'s office, we have your test results and the doctor would like to speak with you.",
        "Hi, this is {name} from {child}'s school, we need to reach you about an incident today.",
        "Good {time}, this is the emergency department at the hospital, your family member is here and stable.",
        "Hello, I am calling from the pharmacy, there is an interaction issue with your new prescription we need to discuss.",
        "Hi, this is {name} from your dentist's office, we had a last-minute cancellation and can see you sooner.",
        "Good {time}, this is the veterinarian's office, your pet's lab results are in and we should talk about them.",
        "Hello, this is the school nurse, {child} is not feeling well and we would like you to pick them up.",
        "Hi, I am calling from the auto repair shop, we found something during the inspection that needs attention.",
        "Good afternoon, this is {name} from the after-school program, {child} had a minor incident.",
        "Hello, this is your landlord, there is a maintenance emergency in the building that affects your unit.",
        "Hi, this is the hospital billing department, you have a credit on your account we need to discuss.",
        "Good {time}, your insurance agent is calling because your home policy is up for renewal this week.",
        "Hello, this is {name} from the lab, your doctor ordered a follow-up test and we need to schedule it.",
        "Hi, the daycare center is calling because {child} has a slight fever and our policy requires pickup.",
        "Good morning, this is the tow company, we have your vehicle and need to discuss the next steps.",
    ],
    "middles": [
        "Everything is fine overall, but the doctor wants to go over the results with you in person.",
        "{child} is safe and being looked after, but we need your authorization before we can proceed.",
        "There is nothing to panic about, we just want to make sure you are informed.",
        "The situation is under control but we thought it was important to let you know right away.",
        "We tried reaching you earlier but the call went to voicemail.",
        "Could you come in at your earliest convenience so we can discuss this in detail?",
        "We need your verbal consent before we can administer any treatment.",
        "The repair is something we recommend addressing soon but it is not an emergency.",
        "Your prescription is ready but we want to go over the dosage instructions with you first.",
        "We have contacted the appropriate authorities and everything is being handled properly.",
        "Please bring your insurance card when you come to pick up {child}.",
        "The maintenance team is on site and expects to have the issue resolved within a few hours.",
        "We can schedule a follow-up appointment at a time that works for you.",
        "The inspection revealed normal wear and tear, nothing unexpected for your vehicle's mileage.",
        "We want to make sure you have all the information you need to make a decision.",
    ],
    "closers": [
        "Please give us a call back at {phone} when you have a moment.",
        "We will be here until 5 PM if you would like to come in today.",
        "Thank you, and please do not hesitate to reach out if you have any questions.",
        "We look forward to seeing you at your earliest convenience.",
        "Take care, and let us know if there is anything else we can do.",
    ],
}


# ======================================================================
# Transcript generation helpers
# ======================================================================

def _pick(lst: list[str], n: int) -> list[str]:
    """Pick n unique items from lst (capped at list length), fill templates."""
    n = min(n, len(lst))
    return [_fill(t) for t in random.sample(lst, n)]


def _gen_polite_social_engineering() -> list[tuple[str, int, str]]:
    cat = "polite_social_engineering"
    d = POLITE_SOCIAL_ENGINEERING
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 1, cat))
    for s in _pick(d["builds"], random.randint(2, 4)):
        rows.append((s, 1, cat))
    for s in _pick(d["extractions"], random.randint(1, 3)):
        rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


def _gen_gradual_escalation() -> list[tuple[str, int, str]]:
    cat = "gradual_escalation"
    d = GRADUAL_ESCALATION
    rows: list[tuple[str, int, str]] = []
    # Long innocent section first
    for s in _pick(d["innocent_start"], random.randint(2, 4)):
        rows.append((s, 1, cat))
    for s in _pick(d["normal_middle"], random.randint(3, 5)):
        rows.append((s, 1, cat))
    for s in _pick(d["transition"], random.randint(1, 2)):
        rows.append((s, 1, cat))
    for s in _pick(d["scam_reveal"], random.randint(1, 3)):
        rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


def _gen_vague_impersonation() -> list[tuple[str, int, str]]:
    cat = "vague_impersonation"
    d = VAGUE_IMPERSONATION
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 1, cat))
    for s in _pick(d["builds"], random.randint(2, 4)):
        rows.append((s, 1, cat))
    for s in _pick(d["extractions"], random.randint(1, 3)):
        rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


def _gen_emotional_manipulation() -> list[tuple[str, int, str]]:
    cat = "emotional_manipulation"
    d = EMOTIONAL_MANIPULATION
    rows: list[tuple[str, int, str]] = []
    # Randomly pick one of three sub-types
    subtype = random.choice(["grandparent", "romance", "charity"])
    if subtype == "grandparent":
        for s in _pick(d["grandparent_soft"], random.randint(3, 6)):
            rows.append((s, 1, cat))
    elif subtype == "romance":
        for s in _pick(d["romance_build"], random.randint(2, 4)):
            rows.append((s, 1, cat))
        for s in _pick(d["romance_ask"], random.randint(2, 3)):
            rows.append((s, 1, cat))
    else:
        for s in _pick(d["charity_appeal"], random.randint(2, 4)):
            rows.append((s, 1, cat))
        for s in _pick(d["charity_extract"], random.randint(1, 3)):
            rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


def _gen_technical_sophistication() -> list[tuple[str, int, str]]:
    cat = "technical_sophistication"
    d = TECHNICAL_SOPHISTICATION
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 1, cat))
    for s in _pick(d["builds"], random.randint(2, 5)):
        rows.append((s, 1, cat))
    for s in _pick(d["extractions"], random.randint(1, 3)):
        rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


def _gen_legitimate_sounding_scam() -> list[tuple[str, int, str]]:
    cat = "legitimate_sounding_scam"
    d = LEGITIMATE_SOUNDING_SCAM
    rows: list[tuple[str, int, str]] = []
    subtype = random.choice(["appointment", "delivery", "refund"])
    if subtype == "appointment":
        for s in _pick(d["fake_appointment"], random.randint(3, 6)):
            rows.append((s, 1, cat))
    elif subtype == "delivery":
        for s in _pick(d["fake_delivery"], random.randint(3, 6)):
            rows.append((s, 1, cat))
    else:
        for s in _pick(d["fake_refund"], random.randint(3, 6)):
            rows.append((s, 1, cat))
    for s in _pick(d["closers"], random.randint(0, 1)):
        rows.append((s, 1, cat))
    return rows


# --- Hard negatives ---

def _gen_real_fraud_alert() -> list[tuple[str, int, str]]:
    cat = "real_fraud_alert"
    d = HARD_NEGATIVE_REAL_FRAUD_ALERT
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    for s in _pick(d["middles"], random.randint(3, 6)):
        rows.append((s, 0, cat))
    for s in _pick(d["closers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    return rows


def _gen_real_government_contact() -> list[tuple[str, int, str]]:
    cat = "real_government_contact"
    d = HARD_NEGATIVE_REAL_GOVERNMENT
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    for s in _pick(d["middles"], random.randint(3, 6)):
        rows.append((s, 0, cat))
    for s in _pick(d["closers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    return rows


def _gen_real_tech_support() -> list[tuple[str, int, str]]:
    cat = "real_tech_support"
    d = HARD_NEGATIVE_REAL_TECH_SUPPORT
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    for s in _pick(d["middles"], random.randint(3, 6)):
        rows.append((s, 0, cat))
    for s in _pick(d["closers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    return rows


def _gen_real_urgent_legitimate() -> list[tuple[str, int, str]]:
    cat = "real_urgent_legitimate"
    d = HARD_NEGATIVE_REAL_URGENT
    rows: list[tuple[str, int, str]] = []
    for s in _pick(d["openers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    for s in _pick(d["middles"], random.randint(3, 5)):
        rows.append((s, 0, cat))
    for s in _pick(d["closers"], random.randint(1, 2)):
        rows.append((s, 0, cat))
    return rows


# ======================================================================
# Main generation
# ======================================================================

SCAM_GENERATORS = {
    "polite_social_engineering": _gen_polite_social_engineering,
    "gradual_escalation": _gen_gradual_escalation,
    "vague_impersonation": _gen_vague_impersonation,
    "emotional_manipulation": _gen_emotional_manipulation,
    "technical_sophistication": _gen_technical_sophistication,
    "legitimate_sounding_scam": _gen_legitimate_sounding_scam,
}

HARD_NEG_GENERATORS = {
    "real_fraud_alert": _gen_real_fraud_alert,
    "real_government_contact": _gen_real_government_contact,
    "real_tech_support": _gen_real_tech_support,
    "real_urgent_legitimate": _gen_real_urgent_legitimate,
}


def generate_adversarial_dataset(
    scam_per_category: int = 90,
    hard_neg_per_category: int = 55,
    seed: int = 123,
) -> list[tuple[str, int, str]]:
    """Generate adversarial scam transcripts and hard-negative legitimates.

    Parameters
    ----------
    scam_per_category : int
        Number of transcripts per scam category (~90 * 6 = 540 transcripts,
        each producing 5-15 sentences => 500+ scam sentences guaranteed).
    hard_neg_per_category : int
        Number of transcripts per hard-negative category (~55 * 4 = 220
        transcripts => 200+ hard-negative sentences guaranteed).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[str, int, str]]
        Each tuple is (sentence_text, label, category).
    """
    random.seed(seed)
    all_rows: list[tuple[str, int, str]] = []

    # Adversarial scams
    for cat_name, gen_fn in SCAM_GENERATORS.items():
        for _ in range(scam_per_category):
            transcript = gen_fn()
            all_rows.extend(transcript)
        scam_count = sum(1 for _, lbl, c in all_rows if c == cat_name and lbl == 1)
        print(f"  [adversarial scam] {cat_name}: {scam_per_category} transcripts, {scam_count} sentences")

    # Hard negatives
    for cat_name, gen_fn in HARD_NEG_GENERATORS.items():
        for _ in range(hard_neg_per_category):
            transcript = gen_fn()
            all_rows.extend(transcript)
        neg_count = sum(1 for _, lbl, c in all_rows if c == cat_name and lbl == 0)
        print(f"  [hard negative]    {cat_name}: {hard_neg_per_category} transcripts, {neg_count} sentences")

    random.shuffle(all_rows)
    return all_rows


def write_csv(rows: list[tuple[str, int, str]], output_path: str) -> None:
    """Write generated data to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "category"])
        for text, label, category in rows:
            writer.writerow([text, label, category])
    print(f"Wrote {len(rows)} rows to {output_path}")


def print_stats(rows: list[tuple[str, int, str]]) -> None:
    """Print dataset statistics."""
    from collections import Counter

    total = len(rows)
    scam = sum(1 for _, lbl, _ in rows if lbl == 1)
    legit = total - scam

    print(f"\n{'='*60}")
    print("Adversarial Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total sentences:      {total:,}")
    print(f"  Scam sentences:     {scam:,} ({100*scam/total:.1f}%)")
    print(f"  Legit sentences:    {legit:,} ({100*legit/total:.1f}%)")
    print()

    cat_counts: Counter[str] = Counter()
    cat_label_counts: dict[str, dict[int, int]] = {}
    for _, label, cat in rows:
        cat_counts[cat] += 1
        if cat not in cat_label_counts:
            cat_label_counts[cat] = {0: 0, 1: 0}
        cat_label_counts[cat][label] += 1

    print(f"{'Category':<35} {'Total':>8} {'Scam':>8} {'Legit':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(cat_counts.keys()):
        s = cat_label_counts[cat].get(1, 0)
        lg = cat_label_counts[cat].get(0, 0)
        print(f"{cat:<35} {cat_counts[cat]:>8,} {s:>8,} {lg:>8,}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adversarial scam and hard-negative call transcripts."
    )
    parser.add_argument(
        "--output",
        default="data/raw/synthetic_transcripts/adversarial_calls.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--scam-per-cat",
        type=int,
        default=90,
        help="Number of adversarial scam transcripts per category (default: 90)",
    )
    parser.add_argument(
        "--hard-neg-per-cat",
        type=int,
        default=55,
        help="Number of hard-negative transcripts per category (default: 55)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating adversarial training data")
    print("=" * 60)

    rows = generate_adversarial_dataset(
        scam_per_category=args.scam_per_cat,
        hard_neg_per_category=args.hard_neg_per_cat,
        seed=args.seed,
    )

    print_stats(rows)
    write_csv(rows, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
