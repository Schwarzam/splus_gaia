import os
import re
import argparse
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def parse_size(size_str):
    """
    Parse a human-readable size string (e.g. '1.2M', '340K', '123') into bytes.
    Returns an int or None if it can't parse.
    """
    if not size_str:
        return None
    s = size_str.strip().upper()
    try:
        if s.endswith('K'):
            return int(float(s[:-1]) * 1024)
        if s.endswith('M'):
            return int(float(s[:-1]) * 1024**2)
        if s.endswith('G'):
            return int(float(s[:-1]) * 1024**3)
        return int(s)
    except ValueError:
        return None

def download_from_url(base_url, save_to, num_threads, log_file, auth, pattern, max_depth=50):
    if max_depth <= 0:
        print(f"Max recursion depth reached at {base_url}")
        return

    try:
        resp = requests.get(base_url, auth=auth, timeout=60, verify=False)
        resp.raise_for_status()
    except Exception as e:
        msg = f"Failed to fetch {base_url}: {e}"
        print(msg)
        if log_file:
            with open(log_file, 'a') as log:
                log.write(msg + '\n')
        return

    soup = BeautifulSoup(resp.text, 'html.parser')
    os.makedirs(save_to, exist_ok=True)

    tasks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for link in soup.find_all('a'):
            href = link.get('href')
            if href == '../':
                continue

            full_url = base_url.rstrip('/') + '/' + href.lstrip('/')
            save_path = os.path.join(save_to, href.rstrip('/'))

            # pattern filter
            if pattern and not href.endswith('/') and not re.search(pattern, href):
                continue

            # parse the listing-size if available
            sibling = link.next_sibling or ''
            size_bytes = None
            if isinstance(sibling, str):
                parts = sibling.strip().split()
                if parts:
                    size_bytes = parse_size(parts[-1])

            if href.endswith('/'):
                # recurse into directory
                tasks.append(executor.submit(
                    download_from_url,
                    full_url,
                    save_path,
                    num_threads,
                    log_file,
                    auth,
                    pattern,
                    max_depth - 1
                ))
            else:
                # if listing size known and matches local, skip outright
                if size_bytes is not None and os.path.exists(save_path):
                    if os.path.getsize(save_path) == size_bytes:
                        print(f"Skipping {full_url} (already downloaded, size {size_bytes}).")
                        continue

                tasks.append(executor.submit(
                    download_file,
                    full_url,
                    save_path,
                    log_file,
                    auth
                ))

        # wait for completion
        for task in tasks:
            try:
                task.result(timeout=300)
            except Exception as e:
                msg = f"Task failed: {e}"
                print(msg)
                if log_file:
                    with open(log_file, 'a') as log:
                        log.write(msg + '\n')

def download_file(url, save_path, log_file, auth):
    print(f"→ {url}")
    temp_path = save_path + '.part'
    try:
        # 1) HEAD for size
        head = requests.head(url, auth=auth, timeout=30, verify=False)
        head.raise_for_status()
        cl = head.headers.get('content-length')
        total_size = int(cl) if cl and cl.isdigit() else None

        # 2) If we know total_size and a local file matches it, skip now
        if total_size is not None and os.path.exists(save_path):
            if os.path.getsize(save_path) == total_size:
                print(f"   Skipped (already have {total_size} bytes).")
                return

        # 3) Stream GET
        with requests.get(url, stream=True, auth=auth, timeout=150, verify=False) as resp:
            resp.raise_for_status()
            # maybe server gives content-length here
            if total_size is None:
                cl2 = resp.headers.get('content-length')
                total_size = int(cl2) if cl2 and cl2.isdigit() else None

            downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # 4) final verify
            if total_size is not None and downloaded != total_size:
                raise ValueError(f"Expected {total_size} bytes, got {downloaded}")

            os.replace(temp_path, save_path)
            print(f"   Downloaded ({downloaded} bytes).")

    except Exception as e:
        msg = f"Failed {url}: {e}"
        print(msg)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if log_file:
            with open(log_file, 'a') as log:
                log.write(msg + '\n')

def main():
    parser = argparse.ArgumentParser(
        description='Recursively download from an Nginx autoindex URL.'
    )
    parser.add_argument('url', help='Base URL of the autoindex folder')
    parser.add_argument('save_directory', help='Local folder to save into')
    parser.add_argument(
        '--threads', '-t', type=int, default=5,
        help='Concurrent download threads (default: 5)'
    )
    parser.add_argument(
        '--log', '-l',
        help='File to log download errors'
    )
    parser.add_argument('--user', '-u', help='Username for HTTP basic auth')
    parser.add_argument('--password', '-p', help='Password for HTTP basic auth')
    parser.add_argument(
        '--pattern', '-P',
        help='Regex pattern: only download files whose names match'
    )

    args = parser.parse_args()
    os.makedirs(args.save_directory, exist_ok=True)
    log_file = args.log or os.path.join(args.save_directory, 'download.log')
    auth = (args.user, args.password) if args.user and args.password else None

    print("Starting download…")
    download_from_url(
        args.url.rstrip('/') + '/',
        args.save_directory,
        args.threads,
        log_file,
        auth,
        args.pattern
    )
    print("Done.")

if __name__ == '__main__':
    main()
