"""Generate synthetic SMS spam and ham data for training.

Produces a CSV file with columns: text, label, category
  - label: 1 = spam, 0 = ham (legitimate)
  - ~4,800 ham messages and ~700 spam messages (~5,500 total)

Usage
-----
    python training/generate_sms_data.py [--output data/raw/sms_spam/sms_spam.csv]
                                          [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import string
import sys


# ======================================================================
# SPAM TEMPLATES (50+ variations across multiple scam categories)
# ======================================================================

SPAM_TEMPLATES: dict[str, list[str]] = {
    # ---- Prize / lottery scams ----
    "prize_scam": [
        "Congratulations! You've won a ${amount} prize. Call {phone} to claim now!",
        "WINNER! You have been selected to receive ${amount}. Claim at {url}",
        "You've won a ${amount} gift card from {brand}! Reply YES to claim",
        "Congrats! Your number was chosen for a ${amount} cash prize. Text CLAIM to {shortcode}",
        "CONGRATULATIONS {name}! You won ${amount}! Call {phone} immediately to collect",
        "Lucky winner! ${amount} has been deposited to your name. Visit {url} to withdraw",
        "You are our grand prize winner of ${amount}! Act now before it expires! Call {phone}",
        "FREE ${amount} prize awaits you! Simply call {phone} or visit {url}",
        "Your entry has won ${amount}! This is your FINAL notice. Call {phone}",
        "BREAKING: You've been randomly selected for ${amount}! Text WIN to {shortcode}",
        "Amazing news! You won a {brand} shopping spree worth ${amount}! Click {url}",
        "Jackpot alert! ${amount} prize unclaimed. Must respond within 24hrs. Call {phone}",
    ],

    # ---- Account compromise / phishing ----
    "account_phishing": [
        "URGENT: Your {brand} account has been compromised. Click {url} to verify",
        "Your {brand} account will be suspended in 24hrs. Verify now: {url}",
        "Alert: Unusual activity detected on your {brand} account. Confirm identity at {url}",
        "Security notice: Your {brand} password was changed. If not you, visit {url}",
        "IMPORTANT: Your {brand} account requires immediate verification. Go to {url}",
        "{brand} Security: We detected unauthorized access. Secure your account: {url}",
        "Your {brand} account is at risk! Update your information immediately: {url}",
        "WARNING: Multiple failed login attempts on your {brand}. Reset password: {url}",
        "Verify your {brand} identity now or your account will be permanently locked: {url}",
        "Action required: Confirm your {brand} account details within 12 hours at {url}",
        "Someone tried to access your {brand} from a new device. Verify here: {url}",
        "Your {brand} session has expired. Re-authenticate at {url} to avoid suspension",
    ],

    # ---- Package delivery scams ----
    "delivery_scam": [
        "Your package delivery failed. Confirm at {url}",
        "USPS: Your package #{tracking} could not be delivered. Reschedule: {url}",
        "FedEx: Delivery attempt failed for pkg #{tracking}. Update address: {url}",
        "UPS notification: Package held at facility. Pay ${amount} shipping fee: {url}",
        "Your parcel is waiting for delivery. Confirm your address: {url}",
        "Amazon: Your order cannot be shipped. Update payment method: {url}",
        "DHL: Customs fee of ${amount} required for package #{tracking}. Pay here: {url}",
        "Royal Mail: Your item needs a ${amount} fee before delivery. Pay at {url}",
        "Package #{tracking} returned to sender. Redeliver for ${amount}: {url}",
        "ALERT: Your shipment is on hold. Verify delivery details: {url}",
    ],

    # ---- Banking / financial scams ----
    "banking_scam": [
        "ALERT: Suspicious transaction of ${amount} on your account. Call {phone} immediately",
        "{brand} Bank: Your card ending in {last4} has been blocked. Verify: {url}",
        "Urgent: Your bank account has been frozen due to suspicious activity. Call {phone}",
        "Wire transfer of ${amount} pending approval. Confirm at {url} or it will be cancelled",
        "Your credit card was charged ${amount}. If unauthorized, call {phone} now",
        "{brand}: Your account balance is low. Deposit now to avoid fees: {url}",
        "SECURITY ALERT: Unusual ${amount} withdrawal from your account. Verify: {phone}",
        "Your loan of ${amount} has been pre-approved! Apply now: {url}",
        "Congratulations! You qualify for a ${amount} credit line. Apply: {url}",
        "FINAL NOTICE: Your payment of ${amount} is overdue. Pay now to avoid penalties: {url}",
        "Your direct deposit of ${amount} failed. Update banking info: {url}",
        "Unauthorized charge of ${amount} detected. Dispute now: {url}",
    ],

    # ---- Free entry / contest scams ----
    "contest_scam": [
        "Free entry to win ${amount}! Text WIN to {shortcode}",
        "Reply STOP to opt out, or ENTER to win ${amount} in our daily draw!",
        "Exclusive offer: Win a {brand} phone FREE! Enter at {url}",
        "You've been chosen for a chance to win ${amount}! Click here: {url}",
        "FREE iPhone giveaway! Enter now by texting APPLE to {shortcode}",
        "Win a free vacation to {destination}! Reply YES to enter",
        "Limited spots: Win ${amount} daily. Join free: {url}",
        "Be our next winner! ${amount} up for grabs. Text PLAY to {shortcode}",
        "FLASH GIVEAWAY: ${amount} cash. First 100 entries win! Visit {url}",
        "Enter our sweepstakes for ${amount}! No purchase necessary. Text ENTER to {shortcode}",
    ],

    # ---- Subscription / renewal scams ----
    "subscription_scam": [
        "Your {brand} subscription is expiring. Renew now at {url} to avoid interruption",
        "NOTICE: Auto-renewal of ${amount} for {brand} failed. Update payment: {url}",
        "Your {brand} membership will be charged ${amount}. Cancel at {url} if unauthorized",
        "{brand}: Your premium plan renews in 24hrs for ${amount}. To cancel: {url}",
        "Subscription alert: {brand} will charge ${amount} today. Modify: {url}",
        "Final reminder: Your {brand} trial ends today. Upgrade for ${amount}/mo: {url}",
        "WARNING: {brand} subscription lapsed. Reactivate before data loss: {url}",
        "Your antivirus subscription expired! Renew at {url} to stay protected",
        "Netflix billing failed. Update payment to continue service: {url}",
        "Your iCloud storage is full. Upgrade now or lose data: {url}",
    ],

    # ---- Tech support scams ----
    "tech_support_scam": [
        "Microsoft Alert: Your computer has been infected. Call {phone} for support",
        "VIRUS DETECTED on your device! Call {phone} immediately for removal",
        "Apple Security: Your device has been compromised. Contact support: {phone}",
        "ALERT: {count} malware threats found on your phone. Clean now: {url}",
        "Your device is sending spam. Fix immediately by calling {phone}",
        "Windows Defender: Critical threat detected. Update required: {url}",
        "Your phone has been hacked! Protect your data now: {url}",
        "Google Security: Unauthorized app detected. Remove it: {url}",
        "WARNING: Your browsing data is exposed. Secure now: {url}",
        "System alert: Your phone storage is {pct}% infected. Clean: {url}",
    ],

    # ---- Loan / credit offers ----
    "loan_scam": [
        "Pre-approved! Get ${amount} loan at 0% interest. Apply: {url}",
        "Need cash fast? Get up to ${amount} deposited today! Apply: {url}",
        "Bad credit? No problem! ${amount} loan approved. Details: {url}",
        "INSTANT APPROVAL: ${amount} personal loan. No credit check. Apply: {phone}",
        "Debt consolidation: Lower your payments by {pct}%. Call {phone} today",
        "Student loan forgiveness available! Check eligibility: {url}",
        "Cash advance of ${amount} ready for you. Accept now: {url}",
        "Your mortgage rate can be reduced to {pct}%. Refinance: {phone}",
        "Payday loan: Get ${amount} by tomorrow! No paperwork: {url}",
        "Credit repair: Boost your score {count} points. Free consultation: {phone}",
    ],

    # ---- Government impersonation ----
    "government_scam": [
        "IRS NOTICE: You owe ${amount} in back taxes. Pay now to avoid arrest: {phone}",
        "SSA: Your Social Security number has been suspended. Call {phone} immediately",
        "This is the IRS. A warrant has been issued. Call {phone} to resolve",
        "FBI Alert: Your identity was used in a crime. Verify: {phone}",
        "DMV: Your license will be suspended. Renew now: {url}",
        "Government grant of ${amount} approved for you! Claim: {url}",
        "Census Bureau: Complete your required survey or face penalties: {url}",
        "Medicare: Your benefits are about to expire. Renew: {phone}",
        "Tax refund of ${amount} is available. Claim before it expires: {url}",
        "Court notice: Appear or pay ${amount} fine. Details: {url}",
    ],
}


# ======================================================================
# HAM (LEGITIMATE) TEMPLATES (50+ variations across categories)
# ======================================================================

HAM_TEMPLATES: dict[str, list[str]] = {
    # ---- Personal / casual messages ----
    "personal": [
        "Hey, are you free for lunch today?",
        "Happy birthday! Hope you have a great day",
        "Can you pick up milk on the way home?",
        "Miss you! Let's catch up soon",
        "Thanks for dinner last night, it was delicious",
        "Running about 10 minutes late, sorry!",
        "Do you want to grab coffee this afternoon?",
        "Just got home safely, thanks for checking",
        "What time does the movie start tonight?",
        "Good morning! Have a wonderful day",
        "I'll be there in 15 minutes",
        "Can you send me that recipe you mentioned?",
        "The kids had so much fun at the park today",
        "Happy anniversary to you both!",
        "Don't forget to bring sunscreen tomorrow",
        "I left my jacket at your place",
        "Just saw your Instagram post, looks amazing!",
        "Want to go hiking this weekend?",
        "Your mom called, she wants you to call back",
        "I picked up your dry cleaning",
        "Remember we have dinner with the Johnsons at 7",
        "Got your voicemail, calling you back now",
        "Safe travels! Text me when you land",
        "How's the new job going?",
        "Let me know when you're ready and I'll come pick you up",
        "I brought leftovers for you, they're in the fridge",
        "Sorry I missed your call, was in the shower",
        "Can we reschedule to Thursday instead?",
        "Just wanted to say I'm thinking of you",
        "Did you see the game last night? Amazing finish!",
        "I'll be working late tonight, don't wait up",
        "Hey {name}, thanks for helping me move!",
        "Congratulations on the new baby!",
        "So sorry to hear about your grandmother. Sending love",
        "Haha that's hilarious, you always make me laugh",
        "I'm at the store, do we need anything else?",
        "The plumber can come Wednesday at 2pm, does that work?",
        "Just found out I got the promotion!",
        "Movie was great, you should definitely see it",
        "Going to bed early tonight, exhausted from work",
    ],

    # ---- Work / professional messages ----
    "work": [
        "Meeting moved to 3pm in conference room B",
        "Can you review the Q3 report before the meeting?",
        "The client approved the proposal, great work team!",
        "Don't forget the team lunch tomorrow at noon",
        "I'll send you the updated spreadsheet by end of day",
        "Are you available for a quick call at 2pm?",
        "Please submit your timesheet by Friday",
        "The presentation deck is ready for review",
        "New hire orientation starts Monday at 9am",
        "Building maintenance will be in the office tomorrow",
        "FYI the wifi password changed to {word}{number}",
        "Can you cover my shift on Saturday?",
        "Your expense report has been approved",
        "Reminder: all-hands meeting at 10am tomorrow",
        "The project deadline has been extended to next Friday",
        "I shared the Google Doc with you, please add your section",
        "Office will close early at 3pm for the holiday",
        "HR needs your updated emergency contact info",
        "Server maintenance scheduled for tonight 11pm-2am",
        "The new software license keys are in your email",
        "Parking lot B will be closed for repaving next week",
        "Training session moved to room 204",
        "Please join the Zoom call, link in your calendar",
        "Your PTO request for next week has been approved",
        "The annual review forms are due by the 15th",
    ],

    # ---- Appointment / service reminders ----
    "appointment": [
        "Reminder: Dentist appointment tomorrow at 2:30pm",
        "Your car service is scheduled for Monday at 9am",
        "Appointment confirmed with Dr. {name} on {date}",
        "Your haircut is booked for Saturday at 11am",
        "Reminder: Parent-teacher conference Thursday 4pm",
        "Vet appointment for {pet} on {date} at {time}",
        "Your oil change is due. Schedule at your convenience",
        "Pharmacy: Your prescription is ready for pickup",
        "Library: Your reserved book is available for pickup",
        "Your eye exam is scheduled for {date} at {time}",
        "Dentist: Please arrive 15 minutes early for your appointment",
        "Reminder: Annual physical on {date}. Please fast for 12 hours",
        "Your home inspection is confirmed for {date} at {time}",
        "Auto shop: Your car is ready for pickup",
        "Salon appointment confirmed for {date} at {time}",
    ],

    # ---- Delivery / order notifications ----
    "delivery": [
        "Your Uber ride is arriving in 3 minutes",
        "Your DoorDash order is on its way! ETA: 25 min",
        "Amazon: Your order has shipped. Delivery by {date}",
        "Your package was delivered to your front door",
        "Out for delivery: Package arriving today between 2-6pm",
        "FedEx: Your package has been picked up",
        "Instacart: Your shopper is heading to the store",
        "Your Lyft driver is 2 minutes away",
        "Order confirmed! Your {item} will arrive by {date}",
        "Target: Your order is ready for pickup",
        "Walmart: Your grocery order is being prepared",
        "UPS: Delivery scheduled for tomorrow by end of day",
        "Your food is being prepared and will be ready in 20 min",
        "Grubhub: Your driver has picked up your order",
        "Your subscription box has shipped! Tracking: {tracking}",
    ],

    # ---- Bank / financial notifications (legitimate) ----
    "bank_legit": [
        "Your direct deposit of ${amount} has been received",
        "Transaction alert: ${amount} at {store}",
        "Your credit card payment of ${amount} was received. Thank you",
        "Your monthly statement is ready to view in the app",
        "Balance alert: Your checking account is below ${amount}",
        "Your new debit card has been mailed and will arrive in 5-7 days",
        "Your automatic payment of ${amount} to {company} was processed",
        "Mobile deposit of ${amount} has been approved",
        "Reminder: Your credit card payment is due on {date}",
        "Your account has been credited ${amount} for the refund",
        "New payee {name} added to your bill pay",
        "Your savings goal of ${amount} is {pct}% complete",
        "ATM withdrawal of ${amount} at {location}",
        "Your wire transfer of ${amount} has been completed",
        "Interest payment of ${amount} deposited to your savings",
    ],

    # ---- Travel / transport ----
    "travel": [
        "Your flight {flight} departs at {time} from gate {gate}",
        "Check-in is now open for your flight to {destination}",
        "Hotel booking confirmed for {date} at {hotel}",
        "Your rental car is confirmed. Pickup at {location}",
        "Boarding pass ready for {name} on flight {flight}",
        "Flight delay: {flight} now departing at {time}",
        "Your Airbnb host sent you a message",
        "Train tickets confirmed for {date}: {origin} to {destination}",
        "Your travel itinerary has been updated",
        "Airport shuttle departs at {time} from lobby",
    ],

    # ---- School / education ----
    "school": [
        "School is closed tomorrow due to weather",
        "Reminder: Science fair projects are due Friday",
        "Bus route 42 will be 10 minutes late today",
        "Report cards will be available online next week",
        "PTA meeting scheduled for Tuesday at 6pm",
        "Your child's field trip permission slip is due tomorrow",
        "School picture day is next Wednesday",
        "Grades have been posted for the fall semester",
        "Early dismissal on Friday at 1pm",
        "After school tutoring available Mon-Wed 3:30-5pm",
    ],

    # ---- Health / wellness ----
    "health": [
        "Time to take your evening medication",
        "Your lab results are ready. View in the patient portal",
        "Flu shots available at your local pharmacy",
        "Your health insurance card has been mailed",
        "Reminder: Drink water! You haven't logged any today",
        "Your therapy appointment is confirmed for Thursday 3pm",
        "Blood drive this Saturday at the community center",
        "Your prescription will be ready for pickup in 2 hours",
        "Step count today: {count} steps. Keep it up!",
        "Your annual wellness visit is coming up. Schedule at your convenience",
    ],

    # ---- Misc / everyday ----
    "misc": [
        "Weather alert: Rain expected this afternoon",
        "Traffic update: Accident on I-95, expect delays",
        "Your laundry is done. Dryer cycle complete",
        "Power outage reported in your area. Estimated restoration: {time}",
        "Your voter registration has been confirmed",
        "Library books due in 3 days. Renew online or return",
        "Community yard sale this Saturday 8am-2pm",
        "Trash pickup is delayed one day this week due to holiday",
        "Wi-Fi maintenance complete. Network is back online",
        "Your parking meter expires in 15 minutes",
        "Neighborhood watch meeting tonight at 7pm",
        "Water shutoff scheduled for your building on {date} {time}",
        "Your package locker code is {code}. Pickup by {date}",
        "Gas prices dropped! {station} has regular at ${amount}/gal",
        "HOA dues of ${amount} are due by the first of the month",
    ],
}


# ======================================================================
# Random substitution helpers
# ======================================================================

_FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Daniel", "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark",
    "Sandra", "Emily", "Ashley", "Megan", "Brittany", "Samantha", "Lauren",
    "Andrew", "Brian", "Kevin", "Jason", "Timothy", "Steven", "Eric",
]

_BRANDS = [
    "Amazon", "Apple", "Netflix", "PayPal", "Bank of America", "Chase",
    "Wells Fargo", "Citibank", "Microsoft", "Google", "Facebook", "Instagram",
    "Spotify", "Walmart", "Target", "Best Buy", "Costco", "Venmo",
    "Cash App", "T-Mobile", "AT&T", "Verizon", "Sprint", "Comcast",
]

_STORES = [
    "Starbucks", "McDonald's", "Walmart", "Target", "Costco", "Kroger",
    "Whole Foods", "Home Depot", "Lowe's", "CVS", "Walgreens", "Shell",
    "Chevron", "BP", "Trader Joe's", "Publix", "Safeway",
]

_DESTINATIONS = [
    "Hawaii", "Cancun", "Paris", "London", "Tokyo", "Bahamas", "Las Vegas",
    "Orlando", "Miami", "New York", "San Francisco", "Rome", "Barcelona",
]

_HOTELS = [
    "Marriott", "Hilton", "Holiday Inn", "Hyatt", "Best Western",
    "Sheraton", "Westin", "Four Seasons", "Ritz-Carlton",
]

_PETS = ["Max", "Bella", "Charlie", "Luna", "Lucy", "Cooper", "Buddy"]

_ITEMS = [
    "wireless earbuds", "laptop stand", "phone case", "water bottle",
    "backpack", "running shoes", "desk lamp", "keyboard", "monitor",
]

_COMPANIES = [
    "Electric Co", "Gas Company", "Water Utility", "Insurance Inc",
    "Cable Co", "Internet Provider", "Mortgage Corp", "Auto Finance",
]

_LOCATIONS = [
    "Main St ATM", "Airport Terminal", "Downtown Branch", "Mall Location",
    "University Ave", "Park Place", "Highway 101",
]

_STATIONS = ["Shell", "Chevron", "BP", "Mobil", "Sunoco", "Citgo"]


def _random_phone() -> str:
    area = random.randint(200, 999)
    p1 = random.randint(200, 999)
    p2 = random.randint(1000, 9999)
    fmt = random.choice([
        f"({area}) {p1}-{p2}",
        f"{area}-{p1}-{p2}",
        f"+1-{area}-{p1}-{p2}",
        f"1{area}{p1}{p2}",
    ])
    return fmt


def _random_url() -> str:
    """Generate a suspicious-looking URL for spam."""
    suspicious_domains = [
        "bit.ly/{code}", "tinyurl.com/{code}",
        "secure-{brand}-verify.tk/{code}",
        "account-{brand}-login.ml/{code}",
        "{brand}-security-update.ga/verify?id={code}",
        "http://192.168.{a}.{b}:8080/{brand}/login",
        "http://{brand}-verify.com-secure.xyz/{code}",
        "https://www.{brand}-account.suspicious.cf/update",
        "http://{brand}-alert.pw/verify?token={code}",
        "http://secure.{brand}.phish.gq/{code}",
    ]
    template = random.choice(suspicious_domains)
    return template.format(
        code="".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(6, 12))),
        brand=random.choice(_BRANDS).lower().replace(" ", ""),
        a=random.randint(1, 254),
        b=random.randint(1, 254),
    )


def _random_shortcode() -> str:
    return str(random.randint(10000, 99999))


def _random_tracking() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(10, 18)))


def _random_amount() -> str:
    """Generate random dollar amount as string."""
    choice = random.choice(["small", "medium", "large", "huge"])
    if choice == "small":
        val = random.randint(1, 50)
    elif choice == "medium":
        val = random.choice([99, 149, 199, 249, 299, 399, 499])
    elif choice == "large":
        val = random.choice([500, 750, 1000, 1500, 2000, 2500, 3000, 5000])
    else:
        val = random.choice([10000, 25000, 50000, 100000, 500000, 1000000])
    return f"{val:,}"


def _random_date() -> str:
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{month}/{day}"


def _random_time() -> str:
    hour = random.randint(1, 12)
    minute = random.choice(["00", "15", "30", "45"])
    ampm = random.choice(["am", "pm"])
    return f"{hour}:{minute}{ampm}"


def _random_flight() -> str:
    airline = random.choice(["AA", "UA", "DL", "SW", "NK", "B6", "AS"])
    num = random.randint(100, 9999)
    return f"{airline}{num}"


def _random_gate() -> str:
    terminal = random.choice(["A", "B", "C", "D", "E"])
    num = random.randint(1, 50)
    return f"{terminal}{num}"


def _fill_template(template: str) -> str:
    """Replace all placeholders in a template with random values."""
    result = template
    # Replace all placeholders - iterate until no more found
    while "{" in result:
        old = result
        result = result.replace("{name}", random.choice(_FIRST_NAMES), 1)
        result = result.replace("{brand}", random.choice(_BRANDS), 1)
        result = result.replace("{phone}", _random_phone(), 1)
        result = result.replace("{url}", _random_url(), 1)
        result = result.replace("{shortcode}", _random_shortcode(), 1)
        result = result.replace("{amount}", _random_amount(), 1)
        result = result.replace("{tracking}", _random_tracking(), 1)
        result = result.replace("{last4}", str(random.randint(1000, 9999)), 1)
        result = result.replace("{date}", _random_date(), 1)
        result = result.replace("{time}", _random_time(), 1)
        result = result.replace("{store}", random.choice(_STORES), 1)
        result = result.replace("{destination}", random.choice(_DESTINATIONS), 1)
        result = result.replace("{hotel}", random.choice(_HOTELS), 1)
        result = result.replace("{pet}", random.choice(_PETS), 1)
        result = result.replace("{item}", random.choice(_ITEMS), 1)
        result = result.replace("{company}", random.choice(_COMPANIES), 1)
        result = result.replace("{location}", random.choice(_LOCATIONS), 1)
        result = result.replace("{station}", random.choice(_STATIONS), 1)
        result = result.replace("{flight}", _random_flight(), 1)
        result = result.replace("{gate}", _random_gate(), 1)
        result = result.replace("{word}", random.choice(["blue", "green", "coral", "sunset", "alpha", "omega"]), 1)
        result = result.replace("{number}", str(random.randint(100, 999)), 1)
        result = result.replace("{pct}", str(random.randint(10, 90)), 1)
        result = result.replace("{count}", str(random.randint(3, 47)), 1)
        result = result.replace("{code}", str(random.randint(1000, 9999)), 1)
        result = result.replace("{origin}", random.choice(["New York", "Chicago", "Los Angeles", "Boston", "Seattle"]), 1)
        if result == old:
            break  # No more replacements possible
    return result


# ======================================================================
# Generation
# ======================================================================

def generate_sms_data(
    n_ham: int = 4800,
    n_spam: int = 700,
    seed: int = 42,
) -> list[tuple[str, int, str]]:
    """Generate synthetic SMS messages.

    Returns list of (text, label, category) tuples.
    """
    random.seed(seed)
    rows: list[tuple[str, int, str]] = []

    # ---- Generate spam messages ----
    spam_categories = list(SPAM_TEMPLATES.keys())
    # Distribute spam roughly evenly, with some randomness
    per_category = n_spam // len(spam_categories)
    remainder = n_spam % len(spam_categories)

    for i, cat in enumerate(spam_categories):
        templates = SPAM_TEMPLATES[cat]
        count = per_category + (1 if i < remainder else 0)
        for _ in range(count):
            template = random.choice(templates)
            text = _fill_template(template)
            rows.append((text, 1, cat))

    # ---- Generate ham messages ----
    ham_categories = list(HAM_TEMPLATES.keys())
    per_category = n_ham // len(ham_categories)
    remainder = n_ham % len(ham_categories)

    for i, cat in enumerate(ham_categories):
        templates = HAM_TEMPLATES[cat]
        count = per_category + (1 if i < remainder else 0)
        for _ in range(count):
            template = random.choice(templates)
            text = _fill_template(template)
            rows.append((text, 0, cat))

    random.shuffle(rows)
    return rows


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SMS spam/ham data."
    )
    parser.add_argument(
        "--output",
        default="data/raw/sms_spam/sms_spam.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n-ham", type=int, default=4800)
    parser.add_argument("--n-spam", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_sms_data(
        n_ham=args.n_ham,
        n_spam=args.n_spam,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "category"])
        for text, label, category in rows:
            writer.writerow([text, label, category])

    n_spam_actual = sum(1 for _, l, _ in rows if l == 1)
    n_ham_actual = sum(1 for _, l, _ in rows if l == 0)
    print(f"Generated {len(rows):,} SMS messages:")
    print(f"  Ham:  {n_ham_actual:,}")
    print(f"  Spam: {n_spam_actual:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
