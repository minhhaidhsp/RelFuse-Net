"""
One-off diagnostic for the PhysioNet session-login flow used by
download_mimic_subset.py's make_session_auth_context(). Prints ONLY
structural facts (HTTP status codes, whether a CSRF token was found, cookie
NAMES received, any error text PhysioNet's own login form shows, and the
final URL after the login POST) -- never the password, and it strips any
literal password text from printed output as a defensive last resort.

Usage:
    python preprocessing/diagnose_login.py --username hainguyen83
"""
import argparse
import getpass
import http.cookiejar
import os
import re
import sys
import urllib.parse
import urllib.request

PHYSIONET_HOST = "https://physionet.org"


def redact(text, secret):
    return text.replace(secret, "<password>") if secret else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    args = ap.parse_args()
    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass(
        f"PhysioNet password for {args.username}: "
    )

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    ua = {"User-Agent": "diagnose_login.py"}
    login_url = f"{PHYSIONET_HOST}/login/"

    print(f"== GET {login_url} ==")
    get_req = urllib.request.Request(login_url, headers=ua)
    get_resp = opener.open(get_req, timeout=30)
    print("status:", get_resp.status, " final url:", get_resp.geturl())
    html = get_resp.read().decode("utf-8", errors="ignore")
    print("cookies received:", [c.name for c in cj])
    print("html length:", len(html))

    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', html)
    if not m:
        # try the other common attribute order, and a looser fallback, purely diagnostic
        m2 = re.search(r'csrfmiddlewaretoken[^>]*value="([^"]+)"', html)
        print("csrf token (strict order) found:", bool(m), " loose-fallback found:", bool(m2))
        if not m2:
            print("!! No csrfmiddlewaretoken found at all. Dumping any <form> tags found:")
            for f in re.findall(r'<form[^>]*>', html)[:5]:
                print("   ", f)
            sys.exit(1)
        csrf_token = m2.group(1)
    else:
        csrf_token = m.group(1)
    print("csrf_token (first 8 chars):", csrf_token[:8], "... len=", len(csrf_token))

    print(f"\n== POST {login_url} ==")
    data = urllib.parse.urlencode({
        "username": args.username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
    }).encode("utf-8")
    post_req = urllib.request.Request(
        login_url, data=data,
        headers={**ua, "Referer": login_url, "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        post_resp = opener.open(post_req, timeout=30)
        status = post_resp.status
        final_url = post_resp.geturl()
        body = post_resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        status = e.code
        final_url = e.geturl() if hasattr(e, "geturl") else "?"
        body = e.read().decode("utf-8", errors="ignore")

    print("status:", status, " final url:", final_url)
    print("cookies after POST:", [c.name for c in cj])

    body_safe = redact(body, password)
    # Look for common Django auth-form error strings and surface them.
    error_hits = re.findall(r'<[^>]*class="[^"]*error[^"]*"[^>]*>(.*?)</', body_safe, re.IGNORECASE | re.DOTALL)
    if error_hits:
        print("Possible error text found in response:")
        for e in error_hits[:5]:
            print("  -", re.sub(r'\s+', ' ', e).strip()[:300])
    else:
        print("No obvious '...error...' class text found in response body.")

    still_has_csrf_field = 'name="csrfmiddlewaretoken"' in body_safe
    print("Response still contains a login form (csrfmiddlewaretoken input):", still_has_csrf_field)
    print("Response mentions 'Log in' nav link:", "Log in" in body_safe)
    print("Response mentions the username string:", args.username in body_safe)


if __name__ == "__main__":
    main()
