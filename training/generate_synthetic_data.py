"""Generate synthetic scam and legitimate call transcripts for training.

Produces a CSV file with columns: text, label (1=scam, 0=legit), category.
Each row is a single sentence from a generated transcript, enabling
per-sentence fraud scoring during inference.

Usage
-----
    python training/generate_synthetic_data.py [--output data/raw/synthetic_transcripts/calls.csv]
                                                [--scam-per-cat 120]
                                                [--legit-total 1200]
                                                [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Any

# ======================================================================
# SCAM CATEGORIES  (10 categories, 50+ phrases per sub-list)
# ======================================================================

SCAM_CATEGORIES: dict[str, dict[str, list[str]]] = {
    # ------------------------------------------------------------------
    # 1. IRS / Tax Impersonation
    # ------------------------------------------------------------------
    "irs_impersonation": {
        "openers": [
            "This is a final notice from the Internal Revenue Service regarding your federal tax account.",
            "We are calling from the IRS enforcement division about a serious tax violation on your record.",
            "This call is from the United States Treasury Department concerning your outstanding tax liability.",
            "You are receiving this urgent call from the IRS because your tax return has been flagged for fraud.",
            "This is officer {name} from the IRS criminal investigation unit contacting you about unpaid taxes.",
            "Hello, this is the Internal Revenue Service calling to inform you that a federal tax lien has been filed against your name.",
            "We have attempted to reach you multiple times regarding your delinquent tax account with the IRS.",
            "I am calling from the IRS collections department because your tax debt has been escalated.",
            "This message is regarding your overdue tax balance with the IRS which requires immediate resolution.",
            "The Internal Revenue Service has issued a final warning concerning your unpaid federal taxes.",
            "Good afternoon, I am agent {name} badge number {badge} from the IRS audit division.",
            "You have an outstanding balance with the IRS that must be settled before end of business today.",
            "This is an automated call from the IRS to notify you of pending legal action against your tax ID.",
            "I am contacting you on behalf of the IRS regarding case number {case} tied to your Social Security number.",
            "The IRS has been trying to reach you for several months now about a critical tax matter.",
        ],
        "threats": [
            "If you do not resolve this matter immediately, a warrant will be issued for your arrest.",
            "Local police have been notified and they are on their way to your registered address right now.",
            "Your Social Security number will be permanently suspended if payment is not received today.",
            "We will have no choice but to seize your bank accounts, wages, and property.",
            "Failure to pay will result in criminal prosecution and up to five years in federal prison.",
            "A federal arrest warrant has already been signed by the judge and is ready for execution.",
            "Your driver's license and passport will be revoked within the next twenty-four hours.",
            "The sheriff's department will arrive at your home to take you into custody.",
            "All of your assets including your home will be frozen and subject to forfeiture.",
            "You will be placed on a federal watch list and barred from leaving the country.",
            "Legal proceedings have already begun and this is your last chance to avoid a court appearance.",
            "Your employer will be notified of your tax evasion which could result in termination.",
            "The penalties and interest on your account are growing every day you delay.",
            "We will garnish your wages starting next pay period if this is not resolved.",
            "The Department of Justice has been forwarded your case file for criminal tax evasion charges.",
        ],
        "actions": [
            "You need to purchase iTunes gift cards totaling {amount} and read us the redemption codes.",
            "Go to your nearest Walmart or CVS and buy Google Play gift cards for {amount} to settle this debt.",
            "We need you to wire {amount} to our secure government account immediately.",
            "Please provide your bank account number and routing number so we can process a one-time settlement.",
            "You must make an immediate payment of {amount} through Zelle to our collections agent.",
            "Purchase Bitcoin worth {amount} and transfer it to our government wallet address.",
            "Send a wire transfer of {amount} through Western Union to our tax processing center.",
            "We can accept payment through prepaid Visa debit cards totaling {amount}.",
            "You need to transfer {amount} via CashApp to settle your account in full today.",
            "Buy MoneyPak cards totaling {amount} and give us the serial numbers right now.",
            "Provide your credit card information so we can process the settlement payment of {amount}.",
            "Go to the nearest Bitcoin ATM and deposit {amount} using the QR code we will text you.",
            "Send {amount} through Venmo to our authorized collections account.",
            "We accept payment through Amazon gift cards, you need to purchase {amount} worth.",
            "Please stay on the line while you drive to the store to purchase the required gift cards.",
        ],
        "closers": [
            "Do not hang up this phone or the authorities will be dispatched.",
            "Stay on the line and do not contact anyone else about this matter as it is confidential.",
            "This is your final opportunity to resolve this before legal action proceeds.",
            "Remember, this call is being recorded for federal compliance purposes.",
            "Do not discuss this matter with family or friends as it may jeopardize your case.",
        ],
    },

    # ------------------------------------------------------------------
    # 2. Tech Support Scam
    # ------------------------------------------------------------------
    "tech_support": {
        "openers": [
            "This is Microsoft Windows technical support calling about a critical security alert on your computer.",
            "We are calling from the Windows security center because your computer has been compromised by malware.",
            "Hello, this is the Apple support team and we have detected suspicious activity on your iCloud account.",
            "I am calling from Norton antivirus support because your subscription has been used to make unauthorized purchases.",
            "Your computer has been sending us error reports and our technicians have detected a serious virus infection.",
            "This is the Microsoft refund department calling because you were overcharged on your subscription renewal.",
            "We detected that your IP address has been hacked and your personal data is being stolen right now.",
            "Hello, I am from Amazon technical support and someone has accessed your account from a foreign country.",
            "Your antivirus subscription of {amount} has been auto-renewed and we are calling to process your refund.",
            "This is Geek Squad technical support and we need to inform you that your device protection has expired.",
            "We are from the Windows firewall department and your computer has been infected with a trojan virus.",
            "Our security system has detected that your computer is being used to distribute illegal content online.",
            "I am calling from Dell technical support because your warranty is about to expire and your system is at risk.",
            "This is a courtesy call from McAfee security, your computer is broadcasting your passwords to hackers.",
            "Hello, this is the Google security team calling about unauthorized access to your Gmail account.",
        ],
        "threats": [
            "If you do not let us fix this immediately, all your personal files and photos will be permanently deleted.",
            "Hackers are actively downloading your banking information from your computer as we speak.",
            "Your identity is being stolen through your computer and every minute you wait makes it worse.",
            "Without immediate action, your computer will become completely unusable within the next hour.",
            "Criminals are using your computer to commit fraud and you could be held legally responsible.",
            "All your saved passwords, credit card numbers, and personal documents are being copied right now.",
            "Your email account is being used to send phishing emails and you could face legal consequences.",
            "The virus on your computer is spreading to other devices on your network including your phone.",
            "If we do not remove this malware now, your bank accounts will be drained by tonight.",
            "Your webcam has been compromised and hackers are recording you without your knowledge.",
            "This security breach has exposed your Social Security number to criminal organizations.",
            "Your computer has been locked by ransomware and you will lose everything if we do not act now.",
            "Every second counts because the hackers are currently transferring money from your accounts.",
            "Without our intervention your computer will crash permanently and all data will be lost forever.",
            "The malware is encrypting your files one by one and soon nothing will be recoverable.",
        ],
        "actions": [
            "I need you to go to your computer and press the Windows key and the letter R at the same time.",
            "Please download our remote access software from {url} so our technicians can fix the problem.",
            "Type in this web address exactly as I say it so we can connect to your computer securely.",
            "Open your browser and navigate to anydesk dot com to download our remote support tool.",
            "We need you to install TeamViewer so we can access your computer and remove the virus.",
            "Open the command prompt and type the commands I dictate to you to check the error logs.",
            "You need to pay a one-time security fee of {amount} for our premium virus removal service.",
            "We will need your credit card to charge the {amount} technician fee for this emergency repair.",
            "Go to your online banking and transfer {amount} to our certified repair account.",
            "Purchase a {amount} Apple gift card to pay for the lifetime security protection plan.",
            "Please give me the six-digit code that was just sent to your phone so I can verify your identity.",
            "I need you to log into your bank account while I am connected so I can show you the fraudulent charges.",
            "Type eventvwr into the run dialog and I will show you all the error messages proving the hack.",
            "Download the software from supportfix247 dot com and enter the access code I give you.",
            "Let me connect to your machine using the code {code} and I will demonstrate the security breach.",
        ],
        "closers": [
            "Please remain at your computer while our senior technician completes the repairs.",
            "Do not restart your computer or the virus will spread to your other devices.",
            "Stay on the line because disconnecting could cause permanent damage to your system.",
            "Do not tell anyone about this call as it could alert the hackers that we are onto them.",
            "Keep your computer connected to the internet so our tools can finish the scan.",
        ],
    },

    # ------------------------------------------------------------------
    # 3. Bank / Financial Fraud
    # ------------------------------------------------------------------
    "bank_fraud": {
        "openers": [
            "This is the fraud prevention department at your bank calling about suspicious activity on your account.",
            "We detected an unauthorized transaction of {amount} on your debit card ending in {last4}.",
            "Hello, this is your credit card company calling because we noticed unusual spending patterns on your account.",
            "I am calling from Chase Bank security because someone attempted to transfer {amount} from your savings.",
            "Your bank account has been flagged for potential fraudulent activity and we need to verify your identity.",
            "This is Bank of America's fraud alert team and there has been an unauthorized login to your online banking.",
            "We are calling because a wire transfer of {amount} was initiated from your account to an overseas recipient.",
            "Your debit card was used at a location {city} which does not match your usual spending pattern.",
            "Hello, I am from Wells Fargo fraud monitoring and someone has opened a new credit card in your name.",
            "There has been a security breach at your bank and your account information may have been compromised.",
            "This is an urgent call from your bank's security team regarding multiple failed login attempts on your account.",
            "We detected a purchase of {amount} at an electronics store in {city} and need to confirm if this was you.",
            "Your account has been temporarily locked due to suspicious transactions and we need to verify your details.",
            "I am from Citibank's account protection unit calling about a {amount} ATM withdrawal we believe was unauthorized.",
            "This is Capital One calling because your credit card was declined due to suspected fraudulent activity.",
        ],
        "threats": [
            "If we cannot verify your identity your account will be permanently closed within the hour.",
            "Someone is actively draining your account and we need to act right now to stop the transfers.",
            "Your credit score will be severely damaged if these fraudulent charges are not disputed immediately.",
            "The unauthorized user has your full account details and is making purchases every few minutes.",
            "Without verification we will be forced to freeze all your accounts and credit cards.",
            "The criminals have your debit card information and are creating duplicate cards as we speak.",
            "If you do not confirm your identity now, you will be liable for all the fraudulent charges.",
            "Your account is being used for money laundering and you could face legal consequences.",
            "The fraud is escalating rapidly and if we do not stop it now you could lose your entire balance.",
            "Your mortgage payment could bounce if the fraud continues and your home could be at risk.",
            "We have seen cases where victims lose their entire life savings within hours of this type of breach.",
            "Your identity has been sold on the dark web and multiple people are accessing your financial records.",
            "If we cannot secure your account immediately, the bank cannot guarantee your funds.",
            "The unauthorized charges total over {amount} and are growing by the minute.",
            "Without immediate action your auto-pay bills will fail and your credit will be destroyed.",
        ],
        "actions": [
            "I need to verify your identity by confirming your full Social Security number.",
            "Please provide your debit card number, expiration date, and the three-digit security code on the back.",
            "We need your online banking username and password to lock down your account from our end.",
            "Please read me the one-time verification code that was just sent to your phone.",
            "You will need to transfer your funds to a temporary safe account we have set up for you.",
            "Wire {amount} to our secure holding account and we will return it once the investigation is complete.",
            "Provide your PIN number so we can block the card and issue a replacement.",
            "Log into your online banking right now and change your password to what I tell you.",
            "We need to process a temporary reversal by moving your funds to account number {acct}.",
            "Download our secure banking app from the link I will text you to complete the verification.",
            "Please confirm your mother's maiden name and the last four of your Social Security for verification.",
            "Send {amount} via Zelle to our fraud department secure account to process the claim.",
            "Go to the nearest ATM and withdraw {amount} in cash, then deposit it into the safe account I give you.",
            "I need your full debit card number to flag it in our system and prevent further charges.",
            "Verify your identity by providing your date of birth, full name, and billing address.",
        ],
        "closers": [
            "Please do not use your card or online banking until we have completed the investigation.",
            "Do not call the number on the back of your card as that line has been compromised.",
            "Stay on the phone with me while we process this to ensure your account remains secure.",
            "You will receive a confirmation email once we have resolved the situation.",
            "Do not log into your bank account from any other device until we give you the all clear.",
        ],
    },

    # ------------------------------------------------------------------
    # 4. Grandparent Scam
    # ------------------------------------------------------------------
    "grandparent_scam": {
        "openers": [
            "Grandma, it's me, please don't tell mom and dad, I am in serious trouble.",
            "Hi grandpa, this is your grandson, I have been arrested and I need your help right away.",
            "Grandma, I was in a car accident and I am at the hospital, I really need your help.",
            "It's me, your grandchild, I got into a fight and I have been detained by the police.",
            "Grandpa, I am stuck in {city} and I lost my wallet and phone, this is the only call I could make.",
            "Hi nana, I am in jail and they said I can make one phone call, please help me.",
            "Grandma, I was driving and I hit someone and they are saying I was drinking, I am so scared.",
            "Please don't be mad at me grandpa, I got arrested for something I didn't do and I need bail money.",
            "Nana, it's {name}, I am overseas and I was robbed, they took everything, I need money for a flight home.",
            "Grandpa, I broke my nose in the accident so my voice sounds different, please just listen.",
            "Grandma, I am calling from the police station, they are going to charge me if I don't pay.",
            "Hi, this is your grandson's attorney, he has been involved in an incident and needs your help.",
            "Your grandchild asked me to call you because they are not allowed to make calls from detention.",
            "This is officer {name}, I have your grandchild here at the station and they asked me to contact you.",
            "Grandma, please do not tell anyone, I am embarrassed and scared, I just need you to help me.",
        ],
        "threats": [
            "If we do not post bail in the next two hours they are going to transfer me to county jail.",
            "The judge said if I cannot pay the fine today I will have to spend the night in jail.",
            "They are going to charge me with a felony if the settlement is not paid immediately.",
            "I will lose my job if my employer finds out, please help me keep this quiet.",
            "The other driver is threatening to press charges and I could go to prison for years.",
            "My court date is today and without the money I will have a permanent criminal record.",
            "They said they will revoke my visa and deport me if I do not pay the fine right away.",
            "The hospital will not treat my injuries until someone pays the medical bills upfront.",
            "If bail is not posted by tonight I will be moved to a maximum security facility.",
            "My lawyer says the charges will be dropped if we can settle this out of court today.",
            "I am terrified grandma, the other inmates are threatening me and I need to get out.",
            "Without the bail money I could spend weeks in jail waiting for a hearing.",
            "The embassy said they cannot help unless I pay the processing fees first.",
            "They told me I have one chance to resolve this or they are pressing formal charges.",
            "Please hurry, they only gave me a short window to arrange the payment.",
        ],
        "actions": [
            "I need you to go to the store and buy gift cards worth {amount} and call me back with the numbers.",
            "Can you wire {amount} through Western Union to my lawyer's office to cover the bail?",
            "Please go to the bank and withdraw {amount} in cash, a courier will come pick it up.",
            "Send {amount} through MoneyGram to the address I will give you for the bail bondsman.",
            "I need you to send a wire transfer of {amount} to this account for the attorney fees.",
            "Go to Walgreens and buy Google Play gift cards totaling {amount} for the bail bond agent.",
            "Transfer {amount} through Zelle to the number I am going to give you for the settlement.",
            "Buy Bitcoin worth {amount} and send it to the wallet address my lawyer will text you.",
            "Can you put {amount} on a prepaid Visa card and read the card number to my attorney?",
            "Please send {amount} through CashApp to the lawyer handling my case.",
            "I need {amount} in cash, please have it ready and someone from the attorney's office will come get it.",
            "Wire {amount} to this bank account, my lawyer says it's the fastest way to process bail.",
            "Go to the nearest CVS and get MoneyPak cards, I need {amount} total.",
            "Please do not go through the bank teller, use the ATM so no one asks questions.",
            "My attorney will call you in ten minutes with instructions on where to send the {amount}.",
        ],
        "closers": [
            "Please do not tell mom or dad, I do not want them to worry, I will explain everything later.",
            "Promise me you will not tell anyone in the family about this, I am so embarrassed.",
            "Thank you grandma, I love you so much, just please keep this between us.",
            "Remember, my attorney said this has to stay confidential or it could hurt my case.",
            "I will call you back as soon as I am released, just please hurry with the money.",
        ],
    },

    # ------------------------------------------------------------------
    # 5. Crypto / Investment Scam
    # ------------------------------------------------------------------
    "crypto_investment": {
        "openers": [
            "Hello, I am reaching out because you were selected for an exclusive cryptocurrency investment opportunity.",
            "Congratulations, our analysts have identified you as a perfect candidate for our Bitcoin trading platform.",
            "I am a senior investment advisor and I want to share a private opportunity that is making our clients wealthy.",
            "Have you heard about the new cryptocurrency that is about to launch and is guaranteed to double in value?",
            "We have a limited-time investment opportunity that has been returning over 300 percent to our clients.",
            "Hello, I found your profile online and I believe you would be perfect for our elite trading group.",
            "I am calling from a premier cryptocurrency exchange to offer you our VIP investment package.",
            "Our proprietary AI trading algorithm has been generating consistent returns of 50 percent per month.",
            "You were referred to us by a mutual contact who has already made {amount} with our platform.",
            "This is a once in a lifetime opportunity to invest in a new blockchain technology before it goes public.",
            "Our investment fund has never had a losing month and we are now opening it to a select few individuals.",
            "I am an independent financial consultant specializing in cryptocurrency and I have an opportunity you cannot miss.",
            "Would you like to learn how our clients are making {amount} per week with zero trading experience?",
            "We are launching a new DeFi protocol that guarantees minimum returns of 20 percent monthly.",
            "Hello, I represent a group of elite traders who have developed a foolproof crypto trading system.",
        ],
        "threats": [
            "This offer closes in the next 24 hours and once it is gone you will miss out on life-changing returns.",
            "The price is going up rapidly and if you do not invest now you will be left behind.",
            "Our current round of funding is almost full and we can only accept a few more investors today.",
            "Other investors are committing right now and the minimum investment doubles tomorrow.",
            "You will regret not taking this opportunity when you see how much money everyone else makes.",
            "This cryptocurrency is about to be listed on major exchanges and the price will skyrocket overnight.",
            "If you wait even one more day, the entry price will be ten times higher.",
            "People who missed Bitcoin in the early days wished they had listened, do not make the same mistake.",
            "Our platform has a waitlist of thousands and we are giving you priority access right now.",
            "The regulatory window for this investment is closing and after that it will no longer be available.",
            "Every minute you delay is costing you potential profits of hundreds of dollars.",
            "This token is being backed by major tech companies and the announcement drops next week.",
            "Smart investors are already in and the window for maximum returns is shrinking fast.",
            "If you do not secure your position now another investor will take your spot.",
            "The presale price is the lowest it will ever be and we cannot hold your allocation forever.",
        ],
        "actions": [
            "All you need to do is make an initial deposit of {amount} to activate your trading account.",
            "Transfer {amount} in Bitcoin to our secure wallet to start seeing returns within 48 hours.",
            "Send {amount} through wire transfer to our offshore trading fund based in the Cayman Islands.",
            "Download our trading app and fund your account with a minimum of {amount} to get started.",
            "Purchase {amount} worth of Ethereum and send it to the smart contract address I will provide.",
            "Wire {amount} to our brokerage account and our traders will begin investing on your behalf immediately.",
            "Make a deposit of {amount} using Zelle or CashApp to our fund manager's account.",
            "Buy {amount} in USDT on Coinbase and transfer it to our platform wallet address.",
            "Invest just {amount} today and I guarantee you will see at least a 200 percent return within 30 days.",
            "Send {amount} to fund your account and I will personally manage your portfolio.",
            "Purchase Bitcoin from any exchange and send {amount} worth to the address I am about to give you.",
            "Our minimum investment is {amount} but clients who invest more see proportionally higher returns.",
            "Transfer {amount} via bank wire to our institutional trading account to lock in today's price.",
            "Use this referral link to create your account and deposit {amount} to qualify for the bonus.",
            "All we need is a small deposit of {amount} to verify your account and then the real money starts flowing.",
        ],
        "closers": [
            "I will follow up with you in a few days to show you your account growth.",
            "Remember, this is a private opportunity so please do not share it widely.",
            "You are making a very smart decision, welcome to our community of successful investors.",
            "I will send you the account details via text, make sure you act on it today.",
            "Do not discuss the specific returns with your financial advisor as they will try to discourage you.",
        ],
    },

    # ------------------------------------------------------------------
    # 6. Utility Shut-off Scam
    # ------------------------------------------------------------------
    "utility_shutoff": {
        "openers": [
            "This is an emergency notification from your electric company regarding immediate service disconnection.",
            "I am calling from the power company because your account is severely past due and scheduled for shutoff.",
            "Your gas service will be disconnected within the next two hours unless payment is received.",
            "This is {utility} calling to notify you that a disconnection order has been issued for your address.",
            "We are from the water department and your service is about to be terminated due to nonpayment.",
            "Hello, this is the utility company and your electricity will be shut off today unless you pay immediately.",
            "I am an account specialist from your power provider calling about your delinquent balance of {amount}.",
            "This is an urgent message from {utility} regarding the immediate suspension of your utility services.",
            "Your energy account shows a past-due balance of {amount} and a disconnection crew is already dispatched.",
            "We are calling because your utility payment bounced and your service is now scheduled for immediate shutoff.",
            "This is the collections division of your electric company, your account has been in default for 90 days.",
            "Hello, I am from {utility} customer resolutions and this is a final courtesy call before we disconnect.",
            "Your gas meter is scheduled to be removed today due to your outstanding balance.",
            "This is an automated alert that your water service will be terminated in the next 60 minutes.",
            "I am calling from the utility billing department because there is a hold on your account that requires immediate payment.",
        ],
        "threats": [
            "If payment is not received in the next 30 minutes your power will be shut off today.",
            "A technician is already on the way to your home to disconnect your electricity.",
            "Once disconnected there will be an additional reconnection fee of {amount} on top of your balance.",
            "Your service will remain off for a minimum of 72 hours after disconnection regardless of payment.",
            "You will lose all food in your refrigerator and freezer if the power is cut today.",
            "The disconnection crew cannot be recalled once they arrive at your residence.",
            "You will be reported to the credit bureaus which will significantly damage your credit score.",
            "A late fee of {amount} is being added to your account every day the balance remains unpaid.",
            "Your neighbors have already been disconnected for the same reason, you are next on the list.",
            "If the service is disconnected in winter, your pipes could freeze and cause extensive water damage.",
            "Without electricity your home security system will go offline leaving your property unprotected.",
            "The reconnection process takes up to two weeks and requires a deposit of {amount}.",
            "Your account will be sent to an outside collection agency if not paid within the hour.",
            "We cannot guarantee service restoration before the weekend if disconnection happens today.",
            "The longer you wait the higher the total amount becomes due to compounding penalties.",
        ],
        "actions": [
            "You can prevent the shutoff by making an immediate payment of {amount} using a prepaid debit card.",
            "Go to the nearest convenience store and purchase a MoneyPak card for {amount} and call us with the code.",
            "We need you to pay {amount} via gift cards to process the emergency payment and stop disconnection.",
            "Send {amount} through Zelle to our billing department to immediately halt the disconnection order.",
            "Purchase Google Play gift cards totaling {amount} and provide the codes to our billing system.",
            "Wire {amount} through Western Union to our payment processing center to keep your service active.",
            "Make a {amount} payment using Bitcoin through our secure payment portal.",
            "Go to Walmart and buy Visa gift cards worth {amount} and read us the card numbers.",
            "Transfer {amount} using CashApp to our authorized payment account right now.",
            "We can accept an immediate phone payment of {amount} if you provide your bank account and routing numbers.",
            "Purchase iTunes gift cards for {amount} and give us the codes so we can apply it to your balance.",
            "Send {amount} via MoneyGram to our payment center to stop the disconnection crew.",
            "Buy prepaid Visa cards totaling {amount} and give us the card numbers and PINs.",
            "Provide your credit card number and we will process a one-time payment of {amount}.",
            "Pay {amount} through our phone system using the account number I will give you.",
        ],
        "closers": [
            "Please complete the payment while staying on the line so we can confirm and cancel the shutoff.",
            "Once payment is confirmed your account will return to good standing immediately.",
            "Do not contact the regular customer service line as they cannot help with emergency disconnections.",
            "A confirmation number will be provided once the payment clears.",
            "Time is of the essence so please hurry to the store now while I hold the disconnection.",
        ],
    },

    # ------------------------------------------------------------------
    # 7. Vehicle Warranty Extension Scam
    # ------------------------------------------------------------------
    "warranty_extension": {
        "openers": [
            "We are calling about your vehicle's extended warranty which is about to expire.",
            "This is a final reminder that the manufacturer's warranty on your vehicle is ending this month.",
            "Hello, this is the auto warranty department calling about your car's coverage expiration.",
            "I am calling from the vehicle protection center because your bumper-to-bumper coverage has lapsed.",
            "We have been trying to reach you about your car's extended warranty before it expires permanently.",
            "This is your last chance to renew your vehicle's warranty before you lose all coverage.",
            "Our records show that the factory warranty on your {year} {make} is about to run out.",
            "I am an authorized dealer for extended auto warranties and your vehicle qualifies for our premium plan.",
            "We are reaching out because your vehicle identification number shows expired warranty status.",
            "This is a courtesy call from the national auto warranty center about your coverage options.",
            "Hello, your vehicle is no longer protected under the original manufacturer warranty as of this month.",
            "We noticed that your car warranty expired and we want to offer you an exclusive renewal rate.",
            "I am calling from the auto shield protection program about extending your vehicle coverage.",
            "Your vehicle's powertrain warranty is about to expire and repairs could cost you thousands.",
            "This is the vehicle service department following up on the warranty renewal notice we mailed you.",
        ],
        "threats": [
            "Without an extended warranty a single engine repair could cost you over {amount}.",
            "Modern vehicles have complex electronics and a transmission replacement alone is {amount}.",
            "Once your warranty expires today we cannot guarantee this rate will be available again.",
            "Your vehicle is at the perfect mileage for coverage but we can only offer this for a limited time.",
            "Driving without warranty coverage is like driving without insurance, one breakdown could be devastating.",
            "The average out-of-pocket repair bill for your vehicle model is {amount} per year.",
            "If your engine fails tomorrow you would be looking at a repair bill of {amount} or more.",
            "This special pricing expires at the end of business today and the rate goes up by 40 percent tomorrow.",
            "Every day you drive without coverage is a gamble with your financial security.",
            "Other customers with your vehicle model have reported expensive failures right after their warranty ended.",
            "Your catalytic converter alone costs {amount} to replace and theft of these parts is on the rise.",
            "Without coverage a major electrical system failure would cost more than the car is worth.",
            "We only have three spots left at this discounted rate before we move to the next pricing tier.",
            "Repair costs have increased 30 percent this year alone and they are only going to keep rising.",
            "A broken timing belt could destroy your engine and leave you stranded on the highway.",
        ],
        "actions": [
            "We can activate your coverage today for a one-time payment of just {amount}.",
            "All we need is a small deposit of {amount} by credit card to lock in your rate.",
            "Provide your credit card information and we will start your coverage immediately.",
            "We can set up monthly payments of {amount} or you can pay the full annual premium today.",
            "Give us your vehicle identification number and payment details to finalize enrollment.",
            "Make a payment of {amount} through Zelle or Venmo to our enrollment department.",
            "Send {amount} via wire transfer to activate your comprehensive coverage plan.",
            "We need your name, address, and credit card number to process your warranty application.",
            "Pay {amount} today by debit card and you will be fully covered starting tomorrow.",
            "Purchase a prepaid card for {amount} and call us back with the card details.",
            "Confirm your vehicle details and provide payment of {amount} to secure your plan.",
            "Your first premium payment of {amount} is due today to activate bumper-to-bumper coverage.",
            "Give me your bank account details and we will set up automatic monthly drafts of {amount}.",
            "We accept all major credit cards, just provide the number and we will process {amount} right away.",
            "Go to our website at {url} and enter your payment information to complete enrollment.",
        ],
        "closers": [
            "Do not wait until something breaks because by then it is too late for coverage.",
            "You will receive your coverage documents by email within 24 hours of payment.",
            "Remember you have a 30-day money-back guarantee so there is absolutely no risk.",
            "Call us back at this number if you change your mind but act fast before spots fill up.",
            "We appreciate you taking the time to protect your vehicle investment.",
        ],
    },

    # ------------------------------------------------------------------
    # 8. Prize / Lottery Notification Scam
    # ------------------------------------------------------------------
    "prize_notification": {
        "openers": [
            "Congratulations, you have been selected as the grand prize winner in our national sweepstakes!",
            "I am calling to inform you that you have won a cash prize of {amount} in our annual lottery drawing.",
            "This is the Publishers Clearing House notification department, you have won our million dollar giveaway!",
            "You are one of three lucky winners selected from millions of entries in our promotional contest.",
            "Hello, we are delighted to inform you that your phone number was randomly selected to win {amount}.",
            "Congratulations, your entry has been drawn as the winner of a brand new {car} and {amount} in cash!",
            "I am calling from the National Sweepstakes Bureau to confirm your prize of {amount}.",
            "You have won a luxury vacation package to {destination} plus {amount} in spending money!",
            "This call is to notify you that you are the lucky winner of our holiday promotion worth {amount}.",
            "Great news, your name was drawn in our international lottery and you have won {amount}!",
            "Congratulations, you have been selected to receive a {amount} gift card from our rewards program.",
            "I am pleased to announce that you are the grand prize winner of {amount} in our annual giveaway.",
            "Your phone number was selected from our database as the lucky winner of {amount} cash!",
            "This is an official notification that you have won a prize valued at over {amount}.",
            "Hello, you have been chosen as the winner of our customer appreciation sweepstakes worth {amount}!",
        ],
        "threats": [
            "You must claim your prize within the next 24 hours or it will be forfeited to an alternate winner.",
            "If you do not confirm your prize today the money will be given to someone else.",
            "Your prize has a strict claiming deadline and we have already extended it twice for you.",
            "The IRS requires us to distribute unclaimed prizes to the government after the deadline.",
            "Other winners have already claimed their prizes and yours is the only one still outstanding.",
            "We cannot hold your prize indefinitely and the claiming window closes at midnight tonight.",
            "If you hang up or fail to follow the claiming process your prize is automatically void.",
            "This is the third time we have tried to contact you and this is absolutely the final attempt.",
            "Failure to claim will result in your name being removed from all future drawings permanently.",
            "The prize pool is being distributed right now and unclaimed funds revert to the organization.",
            "Do not miss out on this once-in-a-lifetime opportunity, winners like this are extremely rare.",
            "We are required by law to find a recipient for this prize before the fiscal year ends.",
            "Your prize is tax-free but only if you claim it before the end of the current tax period.",
            "Every hour that passes reduces the total prize amount due to administrative holding fees.",
            "If not claimed today the prize committee will select a new winner from the backup list.",
        ],
        "actions": [
            "To claim your prize you need to pay a processing fee of {amount} using prepaid gift cards.",
            "A small tax payment of {amount} is required before we can release your winnings.",
            "Purchase Google Play gift cards worth {amount} and provide the codes for our processing department.",
            "Wire {amount} to cover the insurance and shipping costs for your prize delivery.",
            "Send {amount} through Western Union to our prize distribution center to initiate the payout.",
            "Provide your bank account details so we can deposit the {amount} prize directly.",
            "Pay the {amount} customs fee through MoneyGram to clear your prize for delivery.",
            "Buy Amazon gift cards totaling {amount} to cover the handling and verification fees.",
            "Transfer {amount} via CashApp to our authorized prize agent to complete your claim.",
            "We need your Social Security number and date of birth to verify your identity for the prize.",
            "Send {amount} in Bitcoin to our verification wallet to unlock your prize distribution.",
            "Purchase a prepaid Visa card for {amount} and call back with the card number for processing.",
            "Provide your credit card to cover the {amount} federal prize tax withholding.",
            "Go to the nearest Walmart and buy MoneyPak cards for {amount} to cover prize taxes.",
            "Give us your full name, address, and bank routing number so we can set up the wire transfer.",
        ],
        "closers": [
            "Once the fee is processed your prize will be delivered within three to five business days.",
            "Keep this reference number handy and do not share it with anyone else.",
            "We will call you back to confirm delivery once the processing fee has been received.",
            "This prize is confidential until officially claimed, please do not post about it online.",
            "Congratulations again and thank you for being a valued participant in our program.",
        ],
    },

    # ------------------------------------------------------------------
    # 9. Loan / Credit Approval Scam
    # ------------------------------------------------------------------
    "loan_approval": {
        "openers": [
            "Congratulations, you have been pre-approved for a personal loan of up to {amount} at a very low interest rate.",
            "I am calling from a licensed lending institution because your application for a {amount} loan has been approved.",
            "Hello, we reviewed your credit profile and you qualify for our exclusive low-interest loan program.",
            "We are reaching out because you have been selected for a guaranteed loan approval regardless of your credit score.",
            "Your pre-approval for a {amount} loan is ready to be finalized with just a few simple steps.",
            "I am a loan specialist and I want to let you know you qualify for our zero-down-payment loan of {amount}.",
            "This is a courtesy call to inform you that you have been approved for a consolidation loan of {amount}.",
            "We can offer you a {amount} loan at just 2 percent interest rate with no credit check required.",
            "Hello, your credit score qualifies you for our premium lending program with rates as low as 1.9 percent.",
            "We are a federal lending agency and you have been chosen for our guaranteed approval loan program.",
            "I am calling because based on your financial profile, you are eligible for an emergency hardship loan of {amount}.",
            "Our lending partners have approved you for a {amount} loan with same-day funding available.",
            "This is a special offer for a {amount} debt consolidation loan that can lower your monthly payments.",
            "You qualify for our student loan forgiveness program and a supplemental loan of {amount}.",
            "Hello, I am from an FHA-approved lender and your pre-qualification for a {amount} home loan is ready.",
        ],
        "threats": [
            "This pre-approval expires today and if you do not act now the offer will be permanently withdrawn.",
            "Interest rates are going up next week and this is the last day we can guarantee this low rate.",
            "Your credit score allows for this rate today but any changes could disqualify you by tomorrow.",
            "We can only hold this approval for the next few hours before it goes to the next applicant.",
            "If you do not finalize today you will need to go through the full application process again.",
            "Other lenders are going to charge you much higher rates, this is the best deal you will find.",
            "Your financial situation may not qualify you in the future so it is best to lock this in now.",
            "The lending pool for this program is almost depleted and we expect it to close by end of day.",
            "Delaying could result in a hard credit inquiry that lowers your score without getting the loan.",
            "This government-backed program is ending soon and will not be renewed for the next fiscal year.",
            "Without this loan your existing debts will continue to accumulate interest and penalties.",
            "Your pre-approval status will expire and we cannot guarantee you will qualify again.",
            "Every day you wait is costing you money in higher interest on your current debts.",
            "This rate is only available to the first 50 applicants and we are already at 47.",
            "If you miss this window the next available approval date is three months from now.",
        ],
        "actions": [
            "To finalize your loan we need a refundable processing fee of {amount} paid by prepaid card.",
            "Send {amount} via wire transfer to cover the loan origination and documentation fees.",
            "Provide your Social Security number, date of birth, and bank account information for verification.",
            "We need a good faith deposit of {amount} to reserve your loan terms and lock in the rate.",
            "Purchase Google Play gift cards worth {amount} to cover the insurance premium on the loan.",
            "Transfer {amount} through Zelle to our loan processing department to complete your application.",
            "Wire {amount} to our escrow account to begin the disbursement process.",
            "Pay {amount} through Western Union to cover the credit report and background check fees.",
            "Provide your bank login credentials so we can verify your income and deposit history.",
            "Send {amount} in Bitcoin to our secure escrow wallet to guarantee your approval.",
            "We need a {amount} security deposit that will be rolled into your first monthly payment.",
            "Give us your debit card number and we will charge a one-time {amount} application fee.",
            "Make a MoneyGram payment of {amount} to our underwriting department.",
            "Provide payment of {amount} via CashApp for the mandatory loan insurance premium.",
            "Purchase Amazon gift cards for {amount} and give us the claim codes for processing.",
        ],
        "closers": [
            "Once the fee is processed your loan funds will be deposited within 24 hours.",
            "This is a completely legitimate program backed by a federally insured institution.",
            "You will receive your loan agreement by email for your records.",
            "Do not contact other lenders as multiple applications can hurt your credit score.",
            "We look forward to helping you achieve your financial goals.",
        ],
    },

    # ------------------------------------------------------------------
    # 10. Social Security Scam
    # ------------------------------------------------------------------
    "social_security": {
        "openers": [
            "This is an urgent message from the Social Security Administration regarding your account.",
            "We are calling from the SSA because suspicious activity has been detected on your Social Security number.",
            "Your Social Security number has been linked to criminal activity and is about to be suspended.",
            "I am officer {name} badge number {badge} from the Social Security fraud department.",
            "This call is to inform you that your Social Security number has been compromised in a data breach.",
            "The Social Security Administration has determined that your benefits are being claimed by another individual.",
            "Hello, we are the SSA investigation unit and we have discovered multiple identities using your number.",
            "Your Social Security account has been flagged for review due to suspected identity theft.",
            "I am calling from the Office of the Inspector General regarding fraudulent use of your Social Security number.",
            "This is an automated message from the SSA, your Social Security number has been used to open bank accounts in Texas.",
            "We have evidence that your Social Security number was used to apply for credit cards across multiple states.",
            "The SSA has received reports that your number was found in connection with drug trafficking operations.",
            "Your Social Security benefits are being redirected to a fraudulent account and we need your help.",
            "I am an investigator with the SSA and we need to verify your identity to protect your benefits.",
            "This is the Social Security Administration calling because your number was found on the dark web.",
        ],
        "threats": [
            "Your Social Security number will be permanently suspended within the next two hours.",
            "A federal arrest warrant has been issued in your name due to the criminal activity on your number.",
            "If you do not cooperate your bank accounts will be frozen by federal authorities.",
            "You will lose all Social Security benefits permanently if this matter is not resolved today.",
            "Law enforcement agents have been notified and are prepared to take you into custody.",
            "Your credit will be destroyed and you will be unable to get loans, jobs, or housing.",
            "The IRS has been notified and you may face additional tax fraud charges.",
            "Your name has been linked to international money laundering operations.",
            "Failure to verify your identity today will result in all your government benefits being terminated.",
            "A lien will be placed on your property and your assets will be seized by federal agents.",
            "Your driver's license will be suspended and your passport will be revoked immediately.",
            "Criminal charges for identity fraud carry up to 20 years in federal prison.",
            "Without immediate resolution your retirement benefits will be permanently forfeited.",
            "The evidence against you is substantial and cooperation is the only way to avoid prosecution.",
            "Your identity will be flagged in every federal database making it impossible to conduct business.",
        ],
        "actions": [
            "We need to verify your Social Security number, please confirm all nine digits.",
            "Provide your full Social Security number, date of birth, and mother's maiden name for verification.",
            "You need to purchase gift cards worth {amount} and provide the codes to clear your name from the system.",
            "Transfer your savings to a government-protected account to prevent seizure, I will give you the details.",
            "Wire {amount} to our SSA evidence department to pay the fine and avoid prosecution.",
            "We need you to withdraw {amount} from your bank and deposit it into a safe federal holding account.",
            "Send {amount} via Zelle to the SSA fraud resolution office to clear the flags on your number.",
            "Buy prepaid debit cards for {amount} and read us the numbers to process your identity verification.",
            "Go to Bitcoin ATM and purchase {amount} worth of cryptocurrency to pay the settlement fee.",
            "Provide your bank account and routing numbers so we can verify your legitimate transactions.",
            "Purchase MoneyPak cards for {amount} and give us the PINs to restore your Social Security account.",
            "Transfer {amount} through Western Union to the SSA investigation unit for processing.",
            "We need your online banking credentials to trace the unauthorized transactions on your accounts.",
            "Send {amount} through CashApp to our secure government account to finalize your case.",
            "Give us your credit card information to process a {amount} verification fee.",
        ],
        "closers": [
            "Do not hang up or call anyone as that could be seen as obstruction of a federal investigation.",
            "Stay on the line while we process your information and clear the alerts.",
            "This matter is strictly confidential and discussing it could jeopardize the investigation.",
            "You will receive a new Social Security number once this case is resolved.",
            "Thank you for your cooperation, this is the best way to protect yourself from further damage.",
        ],
    },
}

# ======================================================================
# LEGITIMATE CALL CATEGORIES (5 categories)
# ======================================================================

LEGITIMATE_CATEGORIES: dict[str, dict[str, list[str]]] = {
    # ------------------------------------------------------------------
    # 1. Customer Service
    # ------------------------------------------------------------------
    "customer_service": {
        "openers": [
            "Hello, thank you for calling customer support, my name is {name}, how can I help you today?",
            "Good {time_of_day}, welcome to our customer service line, how may I assist you?",
            "Thank you for reaching out to us, I would be happy to help you with your question.",
            "Hi there, you have reached our help desk, what can I do for you today?",
            "Good afternoon, this is {name} from customer relations, how can I make your day better?",
            "Welcome to our support team, I see you have been waiting and I apologize for the delay.",
            "Hello, I am {name} and I will be assisting you today, can you tell me what you need help with?",
            "Thank you for your patience, you are now connected with a customer service representative.",
            "Hi, this is the customer care team, I understand you have a question about your order.",
            "Good morning, thank you for calling, could you please provide your account or order number?",
            "Hello and welcome to our support center, let me pull up your account information.",
            "You have reached the customer experience team, my name is {name}, how can I assist?",
            "Hi there, thanks for holding, I am ready to help you with whatever you need.",
            "Good {time_of_day}, I am {name} from the support team, tell me what brings you in today.",
            "Welcome back, I see you called earlier, let me pick up where we left off.",
        ],
        "middles": [
            "Let me look into that for you, can you please verify the email address on your account?",
            "I completely understand your frustration, let me see what I can do to resolve this for you.",
            "I am going to check our system now, could you please hold for just a moment?",
            "That is a great question, let me transfer you to the department that can best help with that.",
            "I see the issue on your account, it looks like there was a billing error on our end.",
            "I am sorry to hear about that experience, let me look into this and get it corrected right away.",
            "The item you ordered is currently in transit and should arrive by {date}.",
            "I have located your order, let me provide you with the current status and tracking information.",
            "Would you like me to send a replacement or would you prefer a full refund?",
            "I can see that your subscription is set to renew on {date}, would you like to make any changes?",
            "Your account has been updated with the new information you provided.",
            "I understand, let me escalate this to my supervisor to make sure we get this handled properly.",
            "The total for your order comes to {amount} including tax and shipping.",
            "I have applied a {amount} credit to your account as a courtesy for the inconvenience.",
            "Our return policy allows returns within 30 days of purchase with the original receipt.",
            "Your warranty covers this repair so there will be no charge for the service.",
            "I can schedule a technician to come out on {date} between 9am and noon, does that work for you?",
            "Let me check if that item is in stock at a store near you.",
            "Your payment was processed successfully and you should see the charge within two business days.",
            "Is there anything else I can help you with today?",
        ],
        "closers": [
            "Thank you for calling, is there anything else I can assist you with today?",
            "I am glad I could help, please do not hesitate to call us again if you need anything.",
            "Your case number is {case} for your reference, have a wonderful day.",
            "Thank you for your patience, we value your business and hope to see you again soon.",
            "I hope we were able to resolve your issue, please take our brief survey after this call.",
        ],
    },

    # ------------------------------------------------------------------
    # 2. Appointment Reminder
    # ------------------------------------------------------------------
    "appointment_reminder": {
        "openers": [
            "Hello, this is a reminder about your upcoming appointment with Dr. {name} on {date}.",
            "Good {time_of_day}, we are calling to confirm your appointment scheduled for {date} at {time}.",
            "Hi, this is {clinic} calling to remind you about your visit tomorrow at {time}.",
            "This is a courtesy call from {name}'s office to confirm your appointment on {date}.",
            "Hello, we wanted to remind you that your dental cleaning is scheduled for {date} at {time}.",
            "Good morning, this is {clinic} with a friendly reminder about your upcoming procedure on {date}.",
            "Hi there, just calling to remind you about your annual check-up on {date} at {time}.",
            "This is an automated reminder that your appointment at {clinic} is scheduled for {date}.",
            "Hello, we are calling from {name}'s office to confirm your consultation on {date} at {time}.",
            "Good afternoon, this is a reminder that your eye exam is coming up on {date}.",
            "Hi, this is {clinic} calling about your follow-up appointment scheduled for next {day}.",
            "We wanted to make sure you remember your therapy session on {date} at {time}.",
            "Hello, this is {clinic} reminding you of your lab work appointment on {date}.",
            "Good {time_of_day}, your physical therapy session is confirmed for {date} at {time}.",
            "Hi, we are calling to confirm that you are still available for your appointment on {date}.",
        ],
        "middles": [
            "Please remember to bring your insurance card and a valid photo ID.",
            "If you need to reschedule, please call us at least 24 hours in advance.",
            "Please arrive 15 minutes early to complete any necessary paperwork.",
            "There is no need to fast before this appointment, you can eat normally.",
            "Remember to bring a list of all current medications you are taking.",
            "Parking is available in the rear lot and the entrance is on {street}.",
            "We have updated our office hours, we are now open from 8am to 6pm on weekdays.",
            "Your copay for this visit will be {amount} which can be paid at the front desk.",
            "Please wear comfortable clothing as the exam may require some movement.",
            "If you are experiencing any symptoms before your visit, please let us know when you arrive.",
            "We have your referral on file from Dr. {name} so everything is ready for you.",
            "Your previous test results have been reviewed and Dr. {name} would like to discuss them.",
            "Please do not eat or drink anything after midnight the night before your procedure.",
            "Bring any X-rays or imaging from other providers if you have them available.",
            "We accept most major insurance plans and can verify your coverage when you arrive.",
            "Our new patient forms are available online at our website if you would like to fill them out ahead.",
            "The appointment should take approximately {duration} so please plan accordingly.",
            "If you need directions to our office, our address is {address}.",
            "We have a cancellation policy of {amount} for no-shows, just so you are aware.",
            "Your flu shot is also available during this visit if you would like to get one.",
        ],
        "closers": [
            "We look forward to seeing you, have a great day.",
            "Press 1 to confirm your appointment or 2 to request a callback to reschedule.",
            "Please call us at {phone} if you have any questions before your appointment.",
            "Thank you and we will see you on {date}.",
            "If you need to cancel, please let us know as soon as possible so we can offer the slot to another patient.",
        ],
    },

    # ------------------------------------------------------------------
    # 3. Delivery Notification
    # ------------------------------------------------------------------
    "delivery_notification": {
        "openers": [
            "Hello, this is {carrier} calling about your package delivery scheduled for today.",
            "Hi, we are calling to let you know that your order from {store} has shipped.",
            "Good {time_of_day}, this is a notification that your package is out for delivery.",
            "Your FedEx package with tracking number ending in {tracking} will arrive by end of day.",
            "Hello, this is UPS and we need to confirm the delivery address for your package.",
            "Hi, your Amazon order of {item} has been delivered to your front door.",
            "We are calling to let you know that your shipment from {store} is at the local distribution center.",
            "Good afternoon, your furniture delivery is scheduled for {date} between {time} and {time}.",
            "Hello, this is the delivery team for your appliance order, we wanted to confirm the delivery window.",
            "Hi there, your grocery delivery from {store} will arrive in approximately 30 minutes.",
            "This is a notification that your prescription from {pharmacy} is ready for delivery.",
            "Good morning, your online order has been shipped and the estimated delivery date is {date}.",
            "Hello, we are the movers scheduled for your relocation on {date} and wanted to confirm details.",
            "Your meal delivery from {restaurant} is on its way and should arrive in about 20 minutes.",
            "Hi, this is {carrier} and your package has been delayed, the new delivery date is {date}.",
        ],
        "middles": [
            "Your tracking number is {tracking} and you can follow the delivery status online.",
            "The package weighs about {weight} pounds and will require a signature upon delivery.",
            "We will need someone over 18 to sign for the package when it arrives.",
            "If no one is home, we will leave the package at the door or with a neighbor.",
            "You can track your delivery in real time using our app or website.",
            "The delivery window is between {time} and {time}, we will call 30 minutes before arrival.",
            "Please make sure the delivery area is clear and accessible for our driver.",
            "If the address is incorrect, please contact us before the delivery attempt.",
            "We offer free returns within 30 days if you are not satisfied with your purchase.",
            "Your package has cleared customs and is now in domestic transit.",
            "The item was shipped from our warehouse in {city} and is currently en route.",
            "Please have your order confirmation number ready when the driver arrives.",
            "We attempted delivery earlier but no one was available, we will try again tomorrow.",
            "Your package is being held at the local post office and can be picked up anytime.",
            "The delivery includes assembly service which should take about {duration}.",
            "Multiple packages are being delivered, the second one will arrive the following day.",
            "There is no additional charge for the delivery, it is included with your order.",
            "Our driver will call you about 15 minutes before arriving at your location.",
            "If you prefer a different delivery date, we can reschedule at no extra cost.",
            "The items are packed securely but please inspect them upon delivery for any damage.",
        ],
        "closers": [
            "Thank you for your order and please reach out if you have any questions.",
            "We hope you enjoy your purchase, have a wonderful day.",
            "You will receive a confirmation email once the delivery is complete.",
            "Please contact us if there are any issues with your delivery.",
            "We appreciate your business and look forward to serving you again.",
        ],
    },

    # ------------------------------------------------------------------
    # 4. Business Call
    # ------------------------------------------------------------------
    "business_call": {
        "openers": [
            "Hello {name}, this is {caller} from {company}, I am calling about our meeting next week.",
            "Good {time_of_day} {name}, I wanted to follow up on the proposal we sent over on {date}.",
            "Hi {name}, this is {caller} returning your call from earlier today.",
            "Hello, I am calling from {company} about the contract we discussed last month.",
            "Good morning {name}, this is {caller} and I would like to schedule a time to discuss the project.",
            "Hi there, I am reaching out from {company} to talk about our partnership opportunity.",
            "Hello {name}, this is {caller} from accounting and I have a question about invoice {number}.",
            "Good afternoon, I am calling to confirm the details for our conference call on {date}.",
            "Hi {name}, I wanted to touch base about the quarterly report we are preparing.",
            "Hello, this is {caller} from HR and I am calling about the new employee onboarding schedule.",
            "Good morning, this is {caller} from the marketing team, I need your input on the campaign.",
            "Hi {name}, I am following up on the samples we shipped to you last week.",
            "Hello, I am {caller} from IT and I am calling about the system upgrade scheduled for this weekend.",
            "Good {time_of_day}, this is {caller} from the legal department regarding the NDA review.",
            "Hi, I am calling from {company} to discuss the vendor agreement renewal.",
        ],
        "middles": [
            "I wanted to go over the timeline for the project and make sure we are aligned on deliverables.",
            "The budget has been approved and we are ready to move forward with the next phase.",
            "Could you send me the updated spreadsheet by end of day so I can review the numbers?",
            "Our team has reviewed your proposal and we have a few questions before we can proceed.",
            "The client meeting went well and they are interested in moving forward with option B.",
            "I need to get your approval on the purchase order before we can place the order with the vendor.",
            "The board meeting has been moved to {date}, please update your calendar accordingly.",
            "We have finalized the design mockups and they are ready for your review in the shared folder.",
            "The performance review cycle starts next month and I need your team's self-assessments by {date}.",
            "Our Q3 results exceeded expectations and I wanted to discuss the implications for our planning.",
            "I have attached the revised contract with the changes we discussed, please review sections 3 and 7.",
            "The training session for the new software is scheduled for {date} and all team members should attend.",
            "We need to discuss the staffing requirements for the upcoming product launch.",
            "The compliance audit findings were positive overall with just a few minor items to address.",
            "I am putting together the agenda for the annual retreat and would love your input on topics.",
            "Can we schedule a 30-minute call this week to go over the integration requirements?",
            "The vendor has agreed to our pricing terms and is ready to sign the agreement.",
            "I wanted to let you know that the shipment cleared customs and will arrive at the warehouse tomorrow.",
            "Our analytics show a 15 percent increase in engagement since we launched the new feature.",
            "The hiring committee has narrowed it down to three candidates and we need to schedule final interviews.",
        ],
        "closers": [
            "Let me know if you have any questions, I am available all week.",
            "I will send over the meeting notes and action items by end of day.",
            "Thanks for your time, let's touch base again on {date}.",
            "Please review the documents and get back to me when you have a chance.",
            "Have a great rest of your day and I will talk to you soon.",
        ],
    },

    # ------------------------------------------------------------------
    # 5. Personal Call
    # ------------------------------------------------------------------
    "personal_call": {
        "openers": [
            "Hey, it's me, just calling to check in and see how you are doing.",
            "Hi {name}, I was thinking about you and wanted to give you a call.",
            "Good {time_of_day}, I hope I am not catching you at a bad time.",
            "Hey there, I just got home from work and thought I would give you a ring.",
            "Hi, it's {caller}, I heard you were not feeling well and wanted to see how you are.",
            "Hello {name}, long time no talk, how have you been?",
            "Hey, I am calling because I have some exciting news to share with you.",
            "Hi there, I was in the neighborhood and was wondering if you want to grab coffee.",
            "Good evening, I just wanted to say happy birthday and hope you have a great day!",
            "Hey {name}, are you free this weekend? I was thinking we could get together.",
            "Hi, just a quick call to let you know I am running about 15 minutes late.",
            "Hello, I am calling to see if you got my text message from earlier today.",
            "Hey, it's me, I have a funny story to tell you about what happened at work today.",
            "Hi {name}, I was going through old photos and found some great ones from our trip.",
            "Good morning, I just wanted to check if we are still on for dinner tonight.",
        ],
        "middles": [
            "So how has everything been going with you and the family?",
            "I saw the pictures from your vacation, it looked absolutely amazing.",
            "Did you hear about the new restaurant that opened up on Main Street?",
            "The kids are doing great, {child} just started soccer and is loving it.",
            "I finally finished that book you recommended and you were right, it was fantastic.",
            "We are thinking about planning a trip for the holidays, any interest in joining us?",
            "How is the new job treating you? Are you settling in okay?",
            "I made that recipe you shared with me and it turned out really well.",
            "The weather has been so nice lately, we should plan an outdoor activity soon.",
            "Did you end up getting that car you were looking at?",
            "I have been meaning to call you about the party we are planning for next month.",
            "How was the concert last night? I saw some great photos on social media.",
            "We just adopted a puppy and the kids are absolutely thrilled about it.",
            "I wanted to tell you about this great podcast I have been listening to lately.",
            "Have you watched that new show everyone is talking about? It's really good.",
            "Sorry I missed your birthday party, I was out of town that weekend.",
            "My garden is finally starting to bloom and the tomatoes are looking really good this year.",
            "I got a promotion at work and I am really excited about it.",
            "We should start planning mom's surprise party, her birthday is coming up fast.",
            "I was just at the grocery store and ran into an old friend from college.",
        ],
        "closers": [
            "Anyway, I should let you go, but let's catch up soon over lunch.",
            "It was so good talking to you, let's not wait so long next time.",
            "Give my love to the family, talk to you later.",
            "Alright, I will let you get back to what you were doing, take care.",
            "Love you, call me anytime, bye for now.",
        ],
    },
}

# ======================================================================
# Names, amounts, locations, etc. for template filling
# ======================================================================

FIRST_NAMES = [
    "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony",
    "Mark", "Donald", "Steven", "Andrew", "Paul", "Joshua", "Kenneth",
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
    "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson",
    "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "Miami", "Atlanta", "Denver",
    "Seattle", "Boston", "Las Vegas", "Portland", "Detroit", "Memphis",
    "Baltimore", "Charlotte", "San Francisco", "Nashville", "Austin",
    "Columbus", "Indianapolis", "Jacksonville", "Fort Worth", "Oklahoma City",
]

AMOUNTS = [
    "$150", "$200", "$299", "$350", "$450", "$500", "$750", "$899",
    "$1,000", "$1,200", "$1,500", "$2,000", "$2,500", "$3,000", "$3,500",
    "$4,000", "$4,500", "$5,000", "$7,500", "$10,000", "$15,000", "$25,000",
    "$50,000", "$100,000", "$250,000", "$500,000", "$1,000,000",
]

UTILITY_NAMES = [
    "Pacific Gas and Electric", "Duke Energy", "Southern California Edison",
    "Florida Power and Light", "Dominion Energy", "National Grid",
    "Consolidated Edison", "Commonwealth Edison", "Eversource Energy",
    "Xcel Energy", "American Electric Power", "Entergy",
]

CAR_MAKES = [
    "Toyota Camry", "Honda Civic", "Ford F-150", "Chevrolet Silverado",
    "BMW 3 Series", "Mercedes C-Class", "Audi A4", "Lexus RX",
    "Hyundai Tucson", "Kia Sportage", "Nissan Altima", "Subaru Outback",
]

DESTINATIONS = [
    "Hawaii", "the Caribbean", "Cancun", "Europe", "Paris", "London",
    "Bahamas", "Fiji", "Maldives", "Bali", "Alaska cruise", "Mediterranean cruise",
]

STORE_NAMES = ["Amazon", "Walmart", "Target", "Best Buy", "Costco", "Home Depot"]
CARRIER_NAMES = ["UPS", "FedEx", "USPS", "DHL", "Amazon Logistics"]
CLINIC_NAMES = [
    "Greenfield Medical Center", "Sunrise Health Clinic", "Valley Care Medical",
    "Lakeside Family Practice", "Mountain View Health", "Riverside Wellness Center",
]
COMPANY_NAMES = [
    "Acme Corp", "Pinnacle Solutions", "Vertex Industries", "Horizon Tech",
    "Summit Global", "Apex Consulting", "Catalyst Innovations", "Meridian Partners",
    "Sterling Associates", "Vanguard Systems", "BlueStar Enterprises", "Nexus Group",
]
PHARMACY_NAMES = ["CVS Pharmacy", "Walgreens", "Rite Aid", "Costco Pharmacy"]
RESTAURANT_NAMES = [
    "Olive Garden", "Chili's", "Panera Bread", "Chipotle", "Panda Express",
]

BADGE_NUMBERS = [str(random.randint(10000, 99999)) for _ in range(50)]
CASE_NUMBERS = [f"TC-{random.randint(100000, 999999)}" for _ in range(50)]
LAST4 = [str(random.randint(1000, 9999)) for _ in range(50)]
TRACKING_NUMBERS = [str(random.randint(1000, 9999)) for _ in range(50)]
PHONE_NUMBERS = [
    f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"
    for _ in range(50)
]
DATES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "January 15th", "February 3rd", "March 22nd", "April 10th",
    "May 5th", "June 18th", "July 7th", "August 25th",
    "September 12th", "October 30th", "November 8th", "December 1st",
    "next Monday", "next Friday", "tomorrow", "the day after tomorrow",
    "this coming Thursday", "the 15th of this month", "end of the month",
]
TIMES = [
    "9:00 AM", "9:30 AM", "10:00 AM", "10:15 AM", "11:00 AM",
    "11:30 AM", "12:00 PM", "1:00 PM", "1:30 PM", "2:00 PM",
    "2:30 PM", "3:00 PM", "3:30 PM", "4:00 PM", "4:30 PM",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
STREETS = [
    "Oak Street", "Main Street", "Elm Avenue", "Park Boulevard",
    "Center Drive", "First Street", "Maple Road", "Pine Lane",
]
DURATIONS = [
    "30 minutes", "45 minutes", "one hour", "an hour and a half", "two hours",
]
CHILD_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Logan", "Mia", "Lucas", "Charlotte", "Aiden", "Amelia",
]
INVOICE_NUMBERS = [f"INV-{random.randint(10000, 99999)}" for _ in range(50)]
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
REMOTE_URLS = [
    "fixmypc247.com/support", "quickremote.net/connect", "securehelp.online/fix",
    "windowsfix.support/download", "pcrepair-now.com/tool",
]
ACCESS_CODES = [str(random.randint(100000, 999999)) for _ in range(50)]
ACCOUNT_NUMBERS = [str(random.randint(10000000, 99999999)) for _ in range(50)]


def _fill_template(template: str) -> str:
    """Replace {placeholder} tokens in a template with random values."""
    replacements = {
        "{name}": random.choice(FIRST_NAMES),
        "{caller}": random.choice(FIRST_NAMES),
        "{badge}": random.choice(BADGE_NUMBERS),
        "{case}": random.choice(CASE_NUMBERS),
        "{amount}": random.choice(AMOUNTS),
        "{city}": random.choice(CITIES),
        "{utility}": random.choice(UTILITY_NAMES),
        "{car}": random.choice(CAR_MAKES),
        "{make}": random.choice(CAR_MAKES),
        "{year}": random.choice(YEARS),
        "{destination}": random.choice(DESTINATIONS),
        "{store}": random.choice(STORE_NAMES),
        "{carrier}": random.choice(CARRIER_NAMES),
        "{clinic}": random.choice(CLINIC_NAMES),
        "{company}": random.choice(COMPANY_NAMES),
        "{pharmacy}": random.choice(PHARMACY_NAMES),
        "{restaurant}": random.choice(RESTAURANT_NAMES),
        "{last4}": random.choice(LAST4),
        "{tracking}": random.choice(TRACKING_NUMBERS),
        "{phone}": random.choice(PHONE_NUMBERS),
        "{date}": random.choice(DATES),
        "{time}": random.choice(TIMES),
        "{day}": random.choice(DAYS),
        "{street}": random.choice(STREETS),
        "{duration}": random.choice(DURATIONS),
        "{child}": random.choice(CHILD_NAMES),
        "{number}": random.choice(INVOICE_NUMBERS),
        "{time_of_day}": random.choice(["morning", "afternoon", "evening"]),
        "{url}": random.choice(REMOTE_URLS),
        "{code}": random.choice(ACCESS_CODES),
        "{acct}": random.choice(ACCOUNT_NUMBERS),
        "{address}": f"{random.randint(100,9999)} {random.choice(STREETS)}",
        "{weight}": str(random.randint(1, 50)),
        "{item}": random.choice([
            "a wireless keyboard", "running shoes", "a laptop stand",
            "noise-cancelling headphones", "a coffee maker", "a tablet case",
            "a desk lamp", "a backpack", "a water bottle", "a phone charger",
        ]),
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def _generate_scam_transcript(category: str) -> list[tuple[str, int, str]]:
    """Generate a single scam transcript as a list of (sentence, label, category).

    Returns between 5 and 20 sentences forming a realistic scam call.
    """
    templates = SCAM_CATEGORIES[category]
    sentences: list[tuple[str, int, str]] = []

    # Structure: 1-2 openers, 2-6 threats, 2-6 actions, 0-2 closers
    n_openers = random.randint(1, 2)
    n_threats = random.randint(2, 6)
    n_actions = random.randint(2, 6)
    n_closers = random.randint(0, 2)

    # Pick and fill templates
    for tmpl in random.sample(templates["openers"], min(n_openers, len(templates["openers"]))):
        sentences.append((_fill_template(tmpl), 1, category))

    for tmpl in random.sample(templates["threats"], min(n_threats, len(templates["threats"]))):
        sentences.append((_fill_template(tmpl), 1, category))

    for tmpl in random.sample(templates["actions"], min(n_actions, len(templates["actions"]))):
        sentences.append((_fill_template(tmpl), 1, category))

    if "closers" in templates and n_closers > 0:
        for tmpl in random.sample(templates["closers"], min(n_closers, len(templates["closers"]))):
            sentences.append((_fill_template(tmpl), 1, category))

    # Occasionally insert filler sentences a scammer might say
    filler_scam = [
        "Are you still there?",
        "I understand this is concerning but we are here to help you.",
        "Please listen carefully as I explain the situation.",
        "Now I need you to follow my instructions exactly.",
        "This is a very serious matter that requires your immediate attention.",
        "I am trying to help you resolve this so please cooperate.",
        "Can you confirm your name and date of birth for our records?",
        "I know this is stressful but together we can fix this today.",
        "Do not put me on hold, this is time sensitive.",
        "Okay, are you near a computer right now?",
    ]
    n_fillers = random.randint(0, 3)
    for _ in range(n_fillers):
        pos = random.randint(1, max(1, len(sentences) - 1))
        sentences.insert(pos, (random.choice(filler_scam), 1, category))

    return sentences


def _generate_legit_transcript(category: str) -> list[tuple[str, int, str]]:
    """Generate a single legitimate transcript as a list of (sentence, label, category)."""
    templates = LEGITIMATE_CATEGORIES[category]
    sentences: list[tuple[str, int, str]] = []

    # Structure: 1-2 openers, 3-10 middles, 1-2 closers
    n_openers = random.randint(1, 2)
    n_middles = random.randint(3, 10)
    n_closers = random.randint(1, 2)

    for tmpl in random.sample(templates["openers"], min(n_openers, len(templates["openers"]))):
        sentences.append((_fill_template(tmpl), 0, category))

    for tmpl in random.sample(templates["middles"], min(n_middles, len(templates["middles"]))):
        sentences.append((_fill_template(tmpl), 0, category))

    for tmpl in random.sample(templates["closers"], min(n_closers, len(templates["closers"]))):
        sentences.append((_fill_template(tmpl), 0, category))

    # Occasionally insert natural filler
    filler_legit = [
        "Sure, one moment please.",
        "Absolutely, let me check on that.",
        "Of course, happy to help.",
        "No problem at all.",
        "Let me pull that up for you.",
        "Great, thanks for confirming.",
        "I appreciate your patience.",
        "That sounds good to me.",
        "Perfect, let me take care of that.",
        "Right, I understand.",
    ]
    n_fillers = random.randint(0, 4)
    for _ in range(n_fillers):
        pos = random.randint(1, max(1, len(sentences) - 1))
        sentences.insert(pos, (random.choice(filler_legit), 0, category))

    return sentences


def generate_dataset(
    scam_per_category: int = 120,
    legit_total: int = 1200,
    seed: int = 42,
) -> list[tuple[str, int, str]]:
    """Generate the full synthetic dataset.

    Parameters
    ----------
    scam_per_category : int
        Number of scam transcripts to generate per category.
    legit_total : int
        Total number of legitimate transcripts across all categories.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[str, int, str]]
        Each tuple is (sentence_text, label, category).
    """
    random.seed(seed)
    all_rows: list[tuple[str, int, str]] = []

    # Generate scam transcripts
    for cat in SCAM_CATEGORIES:
        for _ in range(scam_per_category):
            transcript = _generate_scam_transcript(cat)
            all_rows.extend(transcript)
        print(f"  [scam] {cat}: {scam_per_category} transcripts generated")

    # Generate legitimate transcripts (evenly across categories)
    legit_cats = list(LEGITIMATE_CATEGORIES.keys())
    per_legit_cat = legit_total // len(legit_cats)
    remainder = legit_total % len(legit_cats)

    for i, cat in enumerate(legit_cats):
        n = per_legit_cat + (1 if i < remainder else 0)
        for _ in range(n):
            transcript = _generate_legit_transcript(cat)
            all_rows.extend(transcript)
        print(f"  [legit] {cat}: {n} transcripts generated")

    # Shuffle at transcript boundaries is implicit since each row is independent
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
    total = len(rows)
    scam = sum(1 for _, l, _ in rows if l == 1)
    legit = total - scam

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total sentences:      {total:,}")
    print(f"  Scam sentences:     {scam:,} ({100*scam/total:.1f}%)")
    print(f"  Legit sentences:    {legit:,} ({100*legit/total:.1f}%)")
    print()

    # Per-category breakdown
    from collections import Counter
    cat_counts: Counter[str] = Counter()
    cat_label_counts: dict[str, dict[int, int]] = {}
    for _, label, cat in rows:
        cat_counts[cat] += 1
        if cat not in cat_label_counts:
            cat_label_counts[cat] = {0: 0, 1: 0}
        cat_label_counts[cat][label] += 1

    print(f"{'Category':<30} {'Total':>8} {'Scam':>8} {'Legit':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(cat_counts.keys()):
        s = cat_label_counts[cat].get(1, 0)
        l = cat_label_counts[cat].get(0, 0)
        print(f"{cat:<30} {cat_counts[cat]:>8,} {s:>8,} {l:>8,}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic scam and legitimate call transcripts."
    )
    parser.add_argument(
        "--output",
        default="data/raw/synthetic_transcripts/calls.csv",
        help="Output CSV path (default: data/raw/synthetic_transcripts/calls.csv)",
    )
    parser.add_argument(
        "--scam-per-cat",
        type=int,
        default=120,
        help="Number of scam transcripts per category (default: 120)",
    )
    parser.add_argument(
        "--legit-total",
        type=int,
        default=1200,
        help="Total number of legitimate transcripts (default: 1200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("Generating synthetic call transcripts...")
    rows = generate_dataset(
        scam_per_category=args.scam_per_cat,
        legit_total=args.legit_total,
        seed=args.seed,
    )
    print_stats(rows)
    write_csv(rows, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
