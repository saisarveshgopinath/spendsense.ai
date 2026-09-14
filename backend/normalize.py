"""
normalize.py — Text normalization for messy Indian UPI transaction strings.

GOAL: turn a raw UPI narration/VPA into a clean (merchant_name, category) pair.
    "UPI-SERVICENOW-NET-1234@okaxis"  ->  ("ServiceNow", "subscription")

WHY THIS NEEDS MORE THAN ONE REGEX:
Different banks write UPI narrations in different shapes. This file handles
the 3 most common shapes seen in Indian bank statements:
  1) UPI/P2M/412312312312/ZOMATO/HDFC BANK/xxxx1234/Order
  2) UPI-SERVICENOW-NET-1234@okaxis
  3) swiggy.instamart@ybl                (a plain VPA, no "UPI" prefix)

WHY REGEX ALONE ISN'T ENOUGH:
A regex can pull out the raw text chunk ("SERVICENOW"), but it can't know
that the real brand is stylised as "ServiceNow" (capital N in the middle).
That's not a pattern-matching problem, it's a "do we recognise this brand"
problem — so after the regex step, we look the cleaned token up in a small
dictionary (MERCHANT_INFO below) to get the correct display name + category.
"""

import re

# Filler words that sometimes ride along with a merchant name in a
# narration but aren't part of the brand itself.
NOISE_SUFFIXES = {
    "NET", "PVT", "LTD", "LIMITED", "PRIVATE", "INDIA",
    "SERVICES", "SERVICE", "PAYMENTS", "PAYMENT", "PAY", "COM", "ONLINE",
}

# The lookup dictionary mentioned above.
# KEY = the cleaned, all-caps, no-spaces version of the merchant token.
# VALUE = (correct display name, spend category)
# --> Add more entries here as you see new merchants in your real dataset.
MERCHANT_INFO = {
    "SERVICENOW":      ("ServiceNow", "subscription"),
    "SWIGGY":          ("Swiggy", "food delivery"),
    "SWIGGYINSTAMART": ("Swiggy Instamart", "groceries"),
    "ZOMATO":          ("Zomato", "food delivery"),
    "EATSURE":         ("EatSure", "food delivery"),
    "PAYTM":           ("Paytm", "wallet"),
    "PHONEPE":         ("PhonePe", "wallet"),
    "BHARATPE":        ("BharatPe", "wallet"),
    "AMAZON":          ("Amazon", "shopping"),
    "FLIPKART":        ("Flipkart", "shopping"),
    "MYNTRA":          ("Myntra", "shopping"),
    "BIGBASKET":       ("BigBasket", "groceries"),
    "BLINKIT":         ("Blinkit", "groceries"),
    "ZEPTO":           ("Zepto", "groceries"),
    "UBER":            ("Uber", "travel"),
    "OLA":             ("Ola", "travel"),
    "IRCTC":           ("IRCTC", "travel"),
    "REDBUS":          ("RedBus", "travel"),
    "NETFLIX":         ("Netflix", "subscription"),
    "SPOTIFY":         ("Spotify", "subscription"),
    "AIRTEL":          ("Airtel", "utilities"),
    "JIO":             ("Jio", "utilities"),
}


def extract_raw_merchant_token(upi_string: str) -> str:
    """
    Step 1 of 2: pull just the "merchant chunk" out of the full raw string.
    We try 3 regex patterns in order, one per shape described up top, and
    use whichever one matches first.
    """
    s = upi_string.strip().upper()

    # SHAPE 1: UPI/TYPE/reference/MERCHANT/BANK/vpa/remarks
    # Read as: "UPI/", then an optional type code (P2M etc.) and a "/",
    # then any non-slash text (the reference number) and a "/",
    # then CAPTURE the merchant text up to the next "/".
    m = re.match(r"^UPI/(?:P2M|P2A|DR|CR)?/?[^/]*/([A-Z0-9 &.]+)/", s)
    if m:
        return m.group(1).strip()

    # SHAPE 2: UPI-MERCHANT-EXTRA-1234@bank
    # Read as: "UPI-", then CAPTURE one or more dash-separated words
    # (the merchant, possibly written as multiple words),
    # then optionally "-<a run of digits>" (an order ID we don't want),
    # then "@".
    m = re.match(r"^UPI-([A-Z0-9]+(?:-[A-Z0-9]+)*?)(?:-\d{2,})?@", s)
    if m:
        return m.group(1).strip()

    # SHAPE 3: a plain VPA with no "UPI" prefix, e.g. paytm-9876543210@ptaxis
    # Read as: CAPTURE everything up to an optional trailing digit run,
    # then "@".
    m = re.match(r"^([A-Z0-9.\-]+?)(?:\d{3,})?@", s)
    if m:
        return m.group(1).strip(".-")

    # None of the 3 shapes matched — hand back the raw string so it's at
    # least visible for manual review instead of silently disappearing.
    return s


def clean_merchant_name(raw_token: str):
    """
    Step 2 of 2: turn the raw token from step 1 into (display_name, category).
    """
    # Split "SERVICENOW-NET-1234" into ["SERVICENOW", "NET", "1234"]
    parts = re.split(r"[-. ]+", raw_token.upper())

    cleaned = []
    for p in parts:
        p = re.sub(r"\d+$", "", p)          # drop trailing digits (order IDs / phone numbers)
        if not p or p in NOISE_SUFFIXES:    # drop empty pieces and filler words
            continue
        if not cleaned or cleaned[-1] != p:  # drop an immediate repeat, e.g. SWIGGY-SWIGGY
            cleaned.append(p)

    key = "".join(cleaned)  # ["SERVICE","NOW"] would join to "SERVICENOW" if ever split that way

    if key in MERCHANT_INFO:
        return MERCHANT_INFO[key]

    # Unknown merchant: best-effort title case, tagged "other" so it's easy
    # to find later and add to MERCHANT_INFO.
    fallback_name = " ".join(p.title() for p in cleaned) if cleaned else raw_token.title()
    return fallback_name, "other"


def clean_transaction(upi_string: str) -> dict:
    """The one function the rest of the project actually calls."""
    token = extract_raw_merchant_token(upi_string)
    merchant, category = clean_merchant_name(token)
    return {"raw": upi_string, "merchant": merchant, "category": category}


if __name__ == "__main__":
    # Quick manual test — run `python3 normalize.py` to see this.
    tests = [
        "UPI-SERVICENOW-NET-1234@okaxis",
        "UPI/P2M/412312312312/ZOMATO/HDFC BANK/xxxx1234/Order",
        "UPI-SWIGGY-SWIGGY@YBL-YESB0YBLUPI-409821112233-PAYMENT FROM PHONE",
        "UPI-BHARATPE09876543210@YESBANK-YESB0YESBNK-409821998877-PAY TO BHARATPE",
        "swiggy.instamart@ybl",
        "paytm-9876543210@ptaxis",
        "amazon@apl",
    ]
    for t in tests:
        result = clean_transaction(t)
        print(f"{t}\n  -> {result['merchant']}  [{result['category']}]\n")
