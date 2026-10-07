import re
import ipaddress
import requests
import numpy as np

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


# =========================================================
# PhiUSIIL - 50 Feature Order
# =========================================================

FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef"
]


# =========================================================
# Helpers
# =========================================================

def safe_ratio(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def normalize_url(url):
    url = str(url).strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return url


def get_tld_probability(tld):
    """
    Approximate TLD legitimacy probability.
    This is NOT the original PhiUSIIL probability table.
    """

    high_trust = {
        "com", "org", "net", "edu", "gov",
        "mil", "int", "co.uk", "ac.uk"
    }

    common = {
        "in", "io", "ai", "co", "me",
        "dev", "app", "tech"
    }

    tld = tld.lower().strip(".")

    if tld in high_trust:
        return 0.90

    if tld in common:
        return 0.75

    return 0.30


def calculate_url_similarity(url, hostname):
    """
    Heuristic URL/domain similarity score.

    Higher value = URL looks more related to its domain.
    """

    url_words = set(
        re.findall(r"[a-zA-Z0-9]+", url.lower())
    )

    domain_words = set(
        re.findall(r"[a-zA-Z0-9]+", hostname.lower())
    )

    if not url_words or not domain_words:
        return 0.0

    intersection = len(
        url_words.intersection(domain_words)
    )

    union = len(
        url_words.union(domain_words)
    )

    return 100.0 * safe_ratio(
        intersection,
        union
    )


def max_character_continuation(url):
    if not url:
        return 0

    maximum = 1

    for match in re.finditer(
        r"(.)\1+",
        url
    ):
        maximum = max(
            maximum,
            len(match.group(0))
        )

    return maximum


# =========================================================
# Main Feature Extractor
# =========================================================

def get_features(url):

    url = normalize_url(url)

    if not url:
        return np.zeros(
            50,
            dtype=float
        )

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    domain = parsed.netloc.lower()

    # =====================================================
    # Basic URL Features
    # =====================================================

    url_length = len(url)

    domain_length = len(hostname)

    try:
        ipaddress.ip_address(hostname)
        is_domain_ip = 1
    except Exception:
        is_domain_ip = 0


    # TLD
    tld = ""

    if "." in hostname:
        tld = hostname.split(".")[-1]

    tld_length = len(tld)

    tld_legitimate_prob = get_tld_probability(
        tld
    )


    # Subdomain
    domain_parts = hostname.split(".")

    no_of_subdomain = max(
        len(domain_parts) - 2,
        0
    )


    # HTTPS
    is_https = (
        1
        if parsed.scheme.lower() == "https"
        else 0
    )


    # =====================================================
    # Character Features
    # =====================================================

    letters = sum(
        c.isalpha()
        for c in url
    )

    digits = sum(
        c.isdigit()
        for c in url
    )

    equals = url.count("=")

    qmark = url.count("?")

    ampersand = url.count("&")


    special_chars = sum(
        1
        for c in url
        if not c.isalnum()
        and c not in "/.:_-"
    )


    letter_ratio = safe_ratio(
        letters,
        url_length
    )

    digit_ratio = safe_ratio(
        digits,
        url_length
    )

    special_ratio = safe_ratio(
        special_chars,
        url_length
    )


    # =====================================================
    # Obfuscation
    # =====================================================

    encoded_chars = re.findall(
        r"%[0-9a-fA-F]{2}",
        url
    )

    no_of_obfuscated_char = len(
        encoded_chars
    )

    has_obfuscation = (
        1
        if no_of_obfuscated_char > 0
        else 0
    )

    obfuscation_ratio = safe_ratio(
        no_of_obfuscated_char,
        url_length
    )


    # =====================================================
    # Character Continuation
    # =====================================================

    max_continuation = max_character_continuation(
        url
    )

    char_continuation_rate = safe_ratio(
        max_continuation,
        url_length
    )


    # =====================================================
    # URL Character Probability
    # =====================================================

    url_char_prob = safe_ratio(
        letters + digits,
        url_length
    )


    # =====================================================
    # URL Similarity
    # =====================================================

    url_similarity_index = calculate_url_similarity(
        url,
        hostname
    )


    # =====================================================
    # Download Web Page
    # =====================================================

    html = ""

    soup = None

    response = None

    try:

        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            },
            allow_redirects=True
        )

        html = response.text

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

    except Exception as e:

        print(
            "Website fetch warning:",
            e
        )


    # =====================================================
    # Page Structure
    # =====================================================

    lines = (
        html.splitlines()
        if html
        else []
    )

    line_of_code = len(lines)

    largest_line_length = max(
        (
            len(line)
            for line in lines
        ),
        default=0
    )


    # =====================================================
    # Title
    # =====================================================

    title = ""

    if soup and soup.title:

        title = soup.title.get_text(
            " ",
            strip=True
        )


    has_title = (
        1
        if title
        else 0
    )


    # =====================================================
    # Domain / Title Similarity
    # =====================================================

    domain_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            hostname.lower()
        )
    )

    title_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            title.lower()
        )
    )


    if domain_words and title_words:

        domain_title_match = (
            len(
                domain_words.intersection(
                    title_words
                )
            )
            /
            len(
                domain_words.union(
                    title_words
                )
            )
        )

    else:

        domain_title_match = 0.0


    # =====================================================
    # URL / Title Similarity
    # =====================================================

    url_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            url.lower()
        )
    )


    if url_words and title_words:

        url_title_match = (
            len(
                url_words.intersection(
                    title_words
                )
            )
            /
            len(
                url_words.union(
                    title_words
                )
            )
        )

    else:

        url_title_match = 0.0


    # =====================================================
    # Favicon
    # =====================================================

    has_favicon = 0

    if soup:

        favicon = soup.find(
            "link",
            rel=lambda x:
                x and
                "icon" in str(x).lower()
        )

        if favicon:
            has_favicon = 1


    # =====================================================
    # Robots.txt
    # =====================================================

    robots = 0

    try:

        robots_url = (
            f"{parsed.scheme}://"
            f"{domain}/robots.txt"
        )

        robots_response = requests.get(
            robots_url,
            timeout=5
        )

        if robots_response.status_code == 200:
            robots = 1

    except Exception:
        pass


    # =====================================================
    # Responsive
    # =====================================================

    is_responsive = 0

    if soup:

        viewport = soup.find(
            "meta",
            attrs={
                "name": re.compile(
                    "viewport",
                    re.I
                )
            }
        )

        if viewport:
            is_responsive = 1


    # =====================================================
    # Redirects
    # =====================================================

    no_of_url_redirect = 0

    no_of_self_redirect = 0

    if response:

        no_of_url_redirect = len(
            response.history
        )

        for history_response in response.history:

            try:

                history_domain = (
                    urlparse(
                        history_response.url
                    ).hostname
                    or ""
                )

                if history_domain == hostname:
                    no_of_self_redirect += 1

            except Exception:
                pass


    # =====================================================
    # Description
    # =====================================================

    has_description = 0

    if soup:

        description = soup.find(
            "meta",
            attrs={
                "name": re.compile(
                    "^description$",
                    re.I
                )
            }
        )

        if description:
            has_description = 1


    # =====================================================
    # Popup
    # =====================================================

    no_of_popup = len(
        re.findall(
            r"window\.open|alert\(|prompt\(|confirm\(",
            html,
            re.I
        )
    )


    # =====================================================
    # iFrame
    # =====================================================

    no_of_iframe = (
        len(
            soup.find_all("iframe")
        )
        if soup
        else 0
    )


    # =====================================================
    # External Form Submit
    # =====================================================

    has_external_form_submit = 0

    if soup:

        for form in soup.find_all("form"):

            action = form.get(
                "action",
                ""
            ).strip()

            if not action:
                continue

            try:

                absolute_action = urljoin(
                    url,
                    action
                )

                action_domain = (
                    urlparse(
                        absolute_action
                    ).hostname
                    or ""
                )

                if (
                    action_domain
                    and
                    action_domain != hostname
                ):

                    has_external_form_submit = 1
                    break

            except Exception:
                pass


    # =====================================================
    # Social Network
    # =====================================================

    social_sites = [
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "pinterest.com",
        "tiktok.com"
    ]


    has_social_net = 1 if any(
        site in html.lower()
        for site in social_sites
    ) else 0


    # =====================================================
    # Submit Button
    # =====================================================

    has_submit_button = 0

    if soup:

        submit = soup.find(
            "input",
            attrs={
                "type": re.compile(
                    "^submit$",
                    re.I
                )
            }
        )

        button = soup.find(
            "button"
        )

        if submit or button:
            has_submit_button = 1


    # =====================================================
    # Hidden Fields
    # =====================================================

    has_hidden_fields = 0

    if soup:

        hidden = soup.find(
            "input",
            attrs={
                "type": re.compile(
                    "^hidden$",
                    re.I
                )
            }
        )

        if hidden:
            has_hidden_fields = 1


    # =====================================================
    # Password Field
    # =====================================================

    has_password_field = 0

    if soup:

        password = soup.find(
            "input",
            attrs={
                "type": re.compile(
                    "^password$",
                    re.I
                )
            }
        )

        if password:
            has_password_field = 1


    # =====================================================
    # Banking / Payment / Crypto
    # =====================================================

    lower_url = url.lower()

    lower_html = html.lower()


    bank_words = [
        "bank",
        "banking",
        "account",
        "netbanking",
        "paypal"
    ]


    pay_words = [
        "payment",
        "pay",
        "checkout",
        "billing",
        "card"
    ]


    crypto_words = [
        "bitcoin",
        "crypto",
        "ethereum",
        "wallet",
        "usdt"
    ]


    bank = 1 if any(
        word in lower_url
        or word in lower_html
        for word in bank_words
    ) else 0


    pay = 1 if any(
        word in lower_url
        or word in lower_html
        for word in pay_words
    ) else 0


    crypto = 1 if any(
        word in lower_url
        or word in lower_html
        for word in crypto_words
    ) else 0


    # =====================================================
    # Copyright
    # =====================================================

    has_copyright_info = (
        1
        if (
            "copyright" in lower_html
            or "©" in html
        )
        else 0
    )


    # =====================================================
    # Images
    # =====================================================

    no_of_image = (
        len(
            soup.find_all("img")
        )
        if soup
        else 0
    )


    # =====================================================
    # CSS
    # =====================================================

    no_of_css = (
        len(
            soup.find_all(
                "link",
                rel="stylesheet"
            )
        )
        if soup
        else 0
    )


    # =====================================================
    # JavaScript
    # =====================================================

    no_of_js = (
        len(
            soup.find_all("script")
        )
        if soup
        else 0
    )


    # =====================================================
    # References
    # =====================================================

    no_of_self_ref = 0

    no_of_empty_ref = 0

    no_of_external_ref = 0


    if soup:

        tags = soup.find_all(
            [
                "a",
                "link",
                "script",
                "img",
                "form"
            ]
        )

        for tag in tags:

            ref = (
                tag.get("href")
                or tag.get("src")
                or tag.get("action")
                or ""
            )

            ref = ref.strip()


            if not ref:

                no_of_empty_ref += 1

                continue


            if ref.startswith("#"):

                no_of_self_ref += 1

                continue


            try:

                absolute_ref = urljoin(
                    url,
                    ref
                )

                ref_parsed = urlparse(
                    absolute_ref
                )

                ref_domain = (
                    ref_parsed.hostname
                    or ""
                )


                if (
                    ref_domain
                    and
                    ref_domain != hostname
                ):

                    no_of_external_ref += 1

                else:

                    no_of_self_ref += 1

            except Exception:

                no_of_self_ref += 1


    # =====================================================
    # Final 50 Features
    # =====================================================

    features = [

        url_length,                       # 1
        domain_length,                    # 2
        is_domain_ip,                     # 3
        url_similarity_index,             # 4
        char_continuation_rate,           # 5
        tld_legitimate_prob,              # 6
        url_char_prob,                    # 7
        tld_length,                       # 8
        no_of_subdomain,                  # 9
        has_obfuscation,                  # 10
        no_of_obfuscated_char,            # 11
        obfuscation_ratio,                # 12
        letters,                          # 13
        letter_ratio,                     # 14
        digits,                           # 15
        digit_ratio,                      # 16
        equals,                           # 17
        qmark,                            # 18
        ampersand,                        # 19
        special_chars,                    # 20
        special_ratio,                    # 21
        is_https,                         # 22
        line_of_code,                     # 23
        largest_line_length,              # 24
        has_title,                        # 25
        domain_title_match,               # 26
        url_title_match,                  # 27
        has_favicon,                      # 28
        robots,                            # 29
        is_responsive,                    # 30
        no_of_url_redirect,               # 31
        no_of_self_redirect,              # 32
        has_description,                  # 33
        no_of_popup,                      # 34
        no_of_iframe,                     # 35
        has_external_form_submit,         # 36
        has_social_net,                   # 37
        has_submit_button,                # 38
        has_hidden_fields,                # 39
        has_password_field,               # 40
        bank,                             # 41
        pay,                              # 42
        crypto,                           # 43
        has_copyright_info,               # 44
        no_of_image,                      # 45
        no_of_css,                        # 46
        no_of_js,                         # 47
        no_of_self_ref,                   # 48
        no_of_empty_ref,                  # 49
        no_of_external_ref               # 50
    ]


    # =====================================================
    # Safety Check
    # =====================================================

    if len(features) != 50:

        raise ValueError(
            f"Expected 50 features, got {len(features)}"
        )


    return np.array(
        features,
        dtype=float
    )